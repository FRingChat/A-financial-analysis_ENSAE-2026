"""
Module fonctions.py
Contient l'ensemble des fonctions utilitaires pour :
- Récupérer les clés d'API (Alpha Vantage, Tiingo) stockées de manière sécurisée sur Vercel (.env.local ou os.environ).
- Récupérer et mettre en cache les données comptables SEC EDGAR (Apple CIK 0000320193).
- Récupérer et mettre en cache les projections/consensus des analystes via Alpha Vantage.
- Récupérer et mettre en cache les cours de bourse quotidiens via Tiingo, et mesurer la
  réaction du cours autour de chaque publication de résultats.
- Aligner temporellement les séries temporelles et calculer les métriques d'erreur de prévision.
- Visualiser graphiquement les résultats réels vs prévisions.
"""

import os
import json
import math
import subprocess
import urllib.error
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================================
# 1. GESTION SÉCURISÉE DE LA CLÉ D'API VIA VERCEL
# ==============================================================================

def load_vercel_env(env_files=(".env.local", ".env", ".env.development.local")) -> None:
    """
    Charge les variables d'environnement synchronisées depuis Vercel.
    Vercel génère par défaut un fichier .env.local via la commande 'vercel env pull'.
    """
    for env_file in env_files:
        if os.path.exists(env_file):
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k not in os.environ:
                            os.environ[k] = v

def pull_vercel_env(target_file: str = ".env.local") -> bool:
    """
    Tente de synchroniser automatiquement les variables d'environnement depuis Vercel
    en utilisant la commande 'npx vercel env pull'.
    Retourne True si l'opération a réussi, False sinon.
    """
    try:
        # 'npx --yes' évite que npx attende une confirmation d'installation (blocage sans terminal)
        cmd = f"npx --yes vercel env pull {target_file} --yes"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and os.path.exists(target_file):
            load_vercel_env((target_file,))
            return True
    except Exception:
        pass
    return False

PLACEHOLDER_KEYS = ("VOTRE_CLE_ICI", "[SENSITIVE]")

def _read_key(var_names) -> str:
    """Première variable d'environnement renseignée parmi var_names (None sinon)."""
    for name in var_names:
        key = os.environ.get(name)
        if key and key not in PLACEHOLDER_KEYS:
            return key
    return None

def get_api_key(var_name: str, aliases: tuple = ()) -> str:
    """
    Récupère une clé d'API stockée sur Vercel (ex: 'ALPHAVANTAGE_API_KEY', 'TIINGO_API_KEY') :
    1. Directement depuis os.environ (environnement Vercel en production / cloud)
    2. Depuis .env.local (fichier généré localement par 'npx vercel env pull')
    3. Depuis .env (fichier local de secours)
    4. Tente une synchronisation automatique si possible (une seule fois)
    aliases : autres noms acceptés pour la même clé (ex. orthographe alternative).
    """
    names = (var_name,) + tuple(aliases)

    # 1. Vérification dans l'environnement système
    key = _read_key(names)
    if key:
        return key

    # 2. Chargement des fichiers d'environnement locaux (.env.local généré par Vercel)
    load_vercel_env()
    key = _read_key(names)
    if key:
        return key

    # 3. Tentative de synchronisation via Vercel CLI si le projet est lié
    print("Tentative de synchronisation des variables d'environnement depuis Vercel...")
    if pull_vercel_env():
        key = _read_key(names)
        if key:
            print("Variables d'environnement synchronisées depuis Vercel avec succès !")
            return key

    # 4. Message explicite d'aide à la synchronisation Vercel
    raise ValueError(
        "\n========================================================================\n"
        f"🔑 CLÉ {var_name} INTROUVABLE EN LOCAL\n"
        "Votre clé est configurée dans votre projet Vercel, mais n'est pas encore\n"
        "synchronisée sur votre machine.\n\n"
        "Pour synchroniser votre clé depuis Vercel en 2 étapes rapides :\n"
        "1. Dans votre terminal, liez votre dossier au projet Vercel :\n"
        "   npx vercel link\n"
        "2. Téléchargez votre variable d'environnement :\n"
        "   npx vercel env pull .env.local\n\n"
        "(Le fichier .env.local est déjà protégé dans .gitignore et ne sera jamais publié)\n"
        f"Alternative : créez vous-même un fichier .env.local contenant {var_name}=votre_cle\n"
        "========================================================================"
    )

def get_alphavantage_key() -> str:
    """Récupère la clé Alpha Vantage (variable ALPHAVANTAGE_API_KEY)."""
    return get_api_key("ALPHAVANTAGE_API_KEY")

def get_tiingo_key() -> str:
    """Récupère la clé Tiingo (variable TIINGO_API_KEY, ou TINGO_API_KEY par tolérance)."""
    return get_api_key("TIINGO_API_KEY", aliases=("TINGO_API_KEY",))


# ==============================================================================
# 2. EXTRACTION ET TRAITEMENT DES DONNÉES SEC EDGAR
# ==============================================================================

def fetch_sec_data(cik: str = "CIK0000320193", cache_file: str = "data_cache/sec_apple.json") -> dict:
    """
    Télécharge et met en cache local les déclarations financières officielles d'Apple auprès de la SEC.
    """
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    url = f"https://data.sec.gov/api/xbrl/companyfacts/{cik}.json"
    headers = {'User-Agent': 'ENSAE-Student contact@ensae.fr'}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data

def extract_sec_metric(facts: dict, taxonomy: str, metric_name: str, frequency: str = 'quarterly') -> pd.DataFrame:
    """
    Extrait une métrique sous forme de DataFrame pandas propre.
    
    Paramètres :
        facts : Dictionnaire des faits comptables SEC
        taxonomy : 'us-gaap' ou 'dei'
        metric_name : Nom XBRL de la métrique (ex: 'EarningsPerShareDiluted', 'NetIncomeLoss')
        frequency : 'quarterly' pour les trimestres (~3 mois) ou 'annual' pour les exercices complets (10-K)

    Une même période apparaît dans plusieurs dépôts (dépôt d'origine, puis colonnes de
    comparaison des rapports suivants) : on garde la ligne du dépôt le plus récent.
    Attention : les colonnes 'fy', 'fp', 'form' et 'filed' décrivent alors CE dépôt
    (ex. un 10-K d'un exercice ultérieur), et non la période elle-même.
    """
    if metric_name not in facts.get(taxonomy, {}):
        raise KeyError(f"Métrique '{metric_name}' non trouvée dans la taxonomie {taxonomy}")
    if frequency not in ('quarterly', 'annual'):
        raise ValueError("frequency doit valoir 'quarterly' ou 'annual'")

    units_dict = facts[taxonomy][metric_name]['units']
    unit_key = list(units_dict.keys())[0]
    df = pd.DataFrame(units_dict[unit_key])

    df['end'] = pd.to_datetime(df['end'])
    df['filed'] = pd.to_datetime(df['filed'])

    # Trimestres : ~3 mois (75 à 105 jours), ce qui élimine les cumuls semestriels ou annuels.
    # Exercices : ~1 an (350 à 380 jours), publiés dans les 10-K.
    if frequency == 'quarterly':
        forms, (min_days, max_days) = ['10-Q', '10-K'], (75, 105)
    else:
        forms, (min_days, max_days) = ['10-K'], (350, 380)

    df = df[df['form'].isin(forms)].copy()
    if 'start' in df.columns:
        df['start'] = pd.to_datetime(df['start'])
        df['duration_days'] = (df['end'] - df['start']).dt.days
        df = df[df['duration_days'].between(min_days, max_days)].copy()

    # Déduplication par période (garde la ligne entière du dépôt le plus récent)
    df = df.sort_values('filed').drop_duplicates('end', keep='last').reset_index(drop=True)

    df['metric_name'] = metric_name
    return df


def label_fiscal_quarters(ends: pd.Series, fiscal_year_end: pd.Timestamp) -> pd.DataFrame:
    """
    Exercice ('fy') et trimestre fiscal ('fp' = Q1..Q4) de chaque fin de trimestre, déduits
    du calendrier à partir d'une fin d'exercice de référence (ex. 27/09/2025 pour Apple).
    L'exercice porte le nom de l'année civile où il se termine (convention d'Apple).
    """
    # Nombre de trimestres (~91,3 jours) séparant chaque fin de trimestre de la référence
    n = ((ends - fiscal_year_end).dt.days / 91.31).round().astype(int)
    return pd.DataFrame({
        'fy': fiscal_year_end.year + (n + 3) // 4,
        'fp': 'Q' + ((n - 1) % 4 + 1).astype(str),
    }, index=ends.index)


def get_split_events(symbol: str = "AAPL") -> pd.Series:
    """
    Renvoie les divisions d'actions (date -> ratio, ex. 4.0 pour un 4:1) à partir
    de la colonne 'splitFactor' des cours Tiingo (cache local data_cache/).
    """
    prices = fetch_tiingo_prices(symbol=symbol)
    splits = prices.loc[prices['splitFactor'] != 1, ['date', 'splitFactor']]
    # Tiingo renvoie parfois 7.000007 au lieu de 7 : on arrondit au ratio entier
    return pd.Series(splits['splitFactor'].round(0).values, index=splits['date'].values)


def split_adjust_per_share(df: pd.DataFrame, splits: pd.Series) -> pd.DataFrame:
    """
    Ramène une métrique par action SEC (ex. BPA) à la base de titres actuelle.

    Une valeur SEC est exprimée dans la base de titres en vigueur à sa date de dépôt
    ('filed') : elle doit donc être divisée par toutes les divisions postérieures à ce dépôt.
    (Une même période peut avoir été retraitée dans un dépôt ultérieur : c'est la date
    du dépôt retenu qui compte, pas la fin de période.)
    Ajoute les colonnes 'val_brut' (valeur publiée) et 'facteur_split'.
    """
    df = df.copy()
    df['val_brut'] = df['val']
    df['facteur_split'] = [
        float(np.prod(splits[splits.index > filed].values)) if len(splits) else 1.0
        for filed in df['filed']
    ]
    df['val'] = df['val_brut'] / df['facteur_split']
    return df


def build_quarterly_eps(
    facts: dict,
    symbol: str = "AAPL",
    metric_name: str = 'EarningsPerShareDiluted',
    taxonomy: str = 'us-gaap'
) -> pd.DataFrame:
    """
    BPA trimestriel complet et comparable dans le temps :
    1. Trimestres publiés (10-Q / 10-K de ~3 mois) via extract_sec_metric.
    2. Ajustement des divisions d'actions (base de titres actuelle, comme Alpha Vantage).
    3. Reconstitution des T4 manquants : le 10-K ne publie souvent que le BPA annuel,
       d'où T4 = BPA annuel - (T1 + T2 + T3). Approximation : le BPA n'est pas
       parfaitement additif (le nombre moyen d'actions varie au cours de l'exercice).
       Ces lignes sont marquées 'q4_derive' = True.
    4. Colonnes 'fy' / 'fp' recalculées à partir des dates (exercice et trimestre fiscal
       de la période), car celles de la SEC décrivent le dépôt retenu (voir extract_sec_metric).
       'form' et 'filed' restent ceux du dépôt retenu.
    """
    splits = get_split_events(symbol)
    quarterly = split_adjust_per_share(
        extract_sec_metric(facts, taxonomy, metric_name, frequency='quarterly'), splits)
    quarterly['q4_derive'] = False

    # Exercices complets (~1 an), dédupliqués par fin de période comme les trimestres
    annual = split_adjust_per_share(
        extract_sec_metric(facts, taxonomy, metric_name, frequency='annual'), splits)

    derived_rows = []
    for _, year in annual.iterrows():
        in_year = quarterly[(quarterly['start'] >= year['start']) & (quarterly['end'] <= year['end'])]
        q4_present = (in_year['end'] - year['end']).abs().min() <= pd.Timedelta('7 days') if len(in_year) else False
        if len(in_year) != 3 or q4_present:
            continue  # T4 déjà publié, ou trimestres T1-T3 incomplets : rien à reconstituer
        q3_end = in_year['end'].max()
        derived_rows.append({
            'end': year['end'],
            'start': q3_end + pd.Timedelta('1 day'),
            'val': year['val'] - in_year['val'].sum(),
            'val_brut': np.nan,
            'facteur_split': year['facteur_split'],
            'accn': year['accn'],
            'fy': year['end'].year,
            'fp': 'Q4',
            'form': '10-K (T4 dérivé)',
            'filed': year['filed'],
            'frame': np.nan,
            'duration_days': (year['end'] - q3_end).days,
            'metric_name': metric_name,
            'q4_derive': True,
        })

    if derived_rows:
        quarterly = pd.concat([quarterly, pd.DataFrame(derived_rows)], ignore_index=True)
    quarterly = quarterly.sort_values('end').reset_index(drop=True)

    # Exercice / trimestre fiscal de chaque période, calés sur la dernière fin d'exercice connue
    quarterly[['fy', 'fp']] = label_fiscal_quarters(quarterly['end'], annual['end'].max())
    return quarterly


# ==============================================================================
# 3. EXTRACTION ET TRAITEMENT DES ESTIMATIONS ALPHA VANTAGE
# ==============================================================================

def fetch_alphavantage_earnings(
    symbol: str = "AAPL",
    cache_dir: str = "data_cache",
    vercel_url: str = "https://projetprog-financeaugustinfelix.vercel.app"
) -> pd.DataFrame:
    """
    Récupère l'historique des prévisions et résultats réels d'Alpha Vantage.
    
    Fonctionnement sans clé pour les utilisateurs externes :
    1. Cache local : Réutilise le fichier local s'il existe (évite toute requête réseau).
    2. Proxy Vercel public : Interroge l'API Vercel qui détient la clé secrète en toute sécurité.
       Une limite stricte de 5 requêtes par jour par utilisateur (IP) est appliquée.
    3. Clé locale (secours) : Si le proxy Vercel n'est pas encore déployé ou accessible,
       utilise la clé locale (.env.local ou .env).
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"av_earnings_{symbol}.json")

    # 1. Utilisation du cache local (0 requête réseau consommée lors des re-exécutions)
    raw_data = None
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if not (isinstance(raw_data, dict) and "quarterlyEarnings" in raw_data):
            print(f"Cache {cache_path} invalide (pas de 'quarterlyEarnings') : nouvelle récupération.")
            raw_data = None

    if raw_data is None:
        proxy_error = None

        # 2. Tentative via le Proxy Vercel (sans clé pour l'utilisateur)
        proxy_endpoint = f"{vercel_url.rstrip('/')}/api/earnings?symbol={symbol}"
        try:
            print(f"Tentative de récupération via le proxy Vercel ({proxy_endpoint})...")
            req = urllib.request.Request(proxy_endpoint, headers={'User-Agent': 'ENSAE-Pandas-Client/1.0'})
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw_data = json.loads(resp.read().decode('utf-8'))

            # Seule une réponse contenant les résultats trimestriels est mise en cache : un message
            # d'erreur (ex. quota Alpha Vantage renvoyé avec un statut 200) bloquerait sinon
            # toutes les exécutions suivantes.
            if isinstance(raw_data, dict) and "quarterlyEarnings" in raw_data:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(raw_data, f, indent=2)
                print("Données récupérées avec succès via l'API Vercel !")
            else:
                proxy_error = f"Réponse inattendue du proxy : {str(raw_data)[:200]}"
                raw_data = None

        except urllib.error.HTTPError as http_err:
            if http_err.code == 429:
                try:
                    err_content = json.loads(http_err.read().decode('utf-8'))
                    msg = err_content.get("message", "Limite de 5 requêtes/jour atteinte.")
                except Exception:
                    msg = "Limite de 5 requêtes/jour atteinte pour votre adresse IP."
                raise PermissionError(f"\n[QUOTA VERCEL DÉPASSÉ] {msg}\n")
            proxy_error = f"HTTP {http_err.code}: {http_err.reason}"
        except Exception as e:
            proxy_error = str(e)

        # 3. Secours : Clé locale si le proxy n'est pas encore déployé
        if raw_data is None:
            print(f"Proxy Vercel non disponible ({proxy_error}). Tentative avec clé locale...")
            try:
                api_key = get_alphavantage_key()
                url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={api_key}"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw_data = json.loads(resp.read().decode('utf-8'))

                if "Error Message" in raw_data:
                    raise ValueError(f"Erreur Alpha Vantage : {raw_data['Error Message']}")
                if "Information" in raw_data or "Note" in raw_data:
                    msg = raw_data.get("Information") or raw_data.get("Note")
                    raise ValueError(f"Quota Alpha Vantage atteint : {msg}")
                if "quarterlyEarnings" not in raw_data:
                    raise ValueError(f"Réponse inattendue d'Alpha Vantage : {str(raw_data)[:200]}")

                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(raw_data, f, indent=2)

            except Exception as fallback_err:
                raise RuntimeError(
                    f"\nImpossible de récupérer les données :\n"
                    f"- Via Proxy Vercel : {proxy_error}\n"
                    f"- Via Clé locale : {fallback_err}\n"
                    f"Assurez-vous d'avoir poussé votre code sur GitHub pour activer l'API Vercel."
                )

    df = pd.DataFrame(raw_data.get("quarterlyEarnings", []))
    if df.empty:
        return df

    df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"])
    df["reportedDate"] = pd.to_datetime(df["reportedDate"])

    numeric_cols = ["reportedEPS", "estimatedEPS", "surprise", "surprisePercentage"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ==============================================================================
# 4. COURS DE BOURSE (TIINGO) ET RÉACTION AUX PUBLICATIONS DE RÉSULTATS
# ==============================================================================

def fetch_tiingo_prices(
    symbol: str = "AAPL",
    start_date: str = "2000-01-01",
    cache_dir: str = "data_cache",
    vercel_url: str = "https://projetprog-financeaugustinfelix.vercel.app",
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Récupère l'historique des cours quotidiens (clôture brute et ajustée) via Tiingo.

    Même logique que fetch_alphavantage_earnings :
    1. Cache local (data_cache/tiingo_prices_<symbol>.json), sauf si force_refresh=True
       (utile pour inclure les séances postérieures au premier téléchargement).
    2. Proxy Vercel public (/api/prices), qui détient la clé TIINGO_API_KEY.
    3. Clé locale (secours) : TIINGO_API_KEY dans .env.local ou .env.

    Une seule requête couvre tout l'historique depuis start_date.
    Colonnes utiles : 'close' (prix réellement coté) et 'adjClose' (ajusté des divisions
    d'actions et des dividendes, comparable dans le temps).
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"tiingo_prices_{symbol}.json")

    raw_data = None
    if os.path.exists(cache_path) and not force_refresh:
        with open(cache_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        proxy_error = None

        # 2. Tentative via le Proxy Vercel (sans clé pour l'utilisateur)
        proxy_endpoint = f"{vercel_url.rstrip('/')}/api/prices?type=prices&symbol={symbol}"
        try:
            print(f"Tentative de récupération via le proxy Vercel ({proxy_endpoint})...")
            req = urllib.request.Request(proxy_endpoint, headers={'User-Agent': 'ENSAE-Pandas-Client/1.0'})
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw_data = json.loads(resp.read().decode('utf-8'))
            if not isinstance(raw_data, list) or not raw_data:
                proxy_error = f"Réponse inattendue du proxy : {str(raw_data)[:200]}"
                raw_data = None
            else:
                print("Cours récupérés avec succès via l'API Vercel !")

        except urllib.error.HTTPError as http_err:
            if http_err.code == 429:
                try:
                    msg = json.loads(http_err.read().decode('utf-8')).get("message", "")
                except Exception:
                    msg = "Limite de 5 requêtes/jour atteinte pour votre adresse IP."
                raise PermissionError(f"\n[QUOTA VERCEL DÉPASSÉ] {msg}\n")
            proxy_error = f"HTTP {http_err.code}: {http_err.reason}"
        except Exception as e:
            proxy_error = str(e)

        # 3. Secours : Clé locale
        if raw_data is None:
            print(f"Proxy Vercel non disponible ({proxy_error}). Tentative avec clé locale...")
            try:
                api_key = get_tiingo_key()
                url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate={start_date}"
                # La clé passe dans l'en-tête (et non dans l'URL) pour ne pas apparaître dans les logs
                req = urllib.request.Request(url, headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Token {api_key}',
                    'User-Agent': 'ENSAE-Pandas-Client/1.0'
                })
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw_data = json.loads(resp.read().decode('utf-8'))

                if isinstance(raw_data, dict):
                    raise ValueError(f"Erreur Tiingo : {raw_data.get('detail', raw_data)}")
                if not raw_data:
                    raise ValueError(f"Aucun cours renvoyé par Tiingo pour {symbol}.")

            except Exception as fallback_err:
                raise RuntimeError(
                    f"\nImpossible de récupérer les cours Tiingo :\n"
                    f"- Via Proxy Vercel : {proxy_error}\n"
                    f"- Via Clé locale : {fallback_err}\n"
                )

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f)

    df = pd.DataFrame(raw_data)
    # Tiingo renvoie des dates du type '2024-08-01T00:00:00.000Z' : on garde la date seule
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def compute_announcement_price_reaction(df_av: pd.DataFrame, df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Mesure la réaction du cours à chaque publication de résultats (fenêtre J -> J+1).

    Apple publie ses résultats APRÈS la clôture de la bourse le jour J (reportedDate) :
    - clôture de J   = dernier prix AVANT l'information
    - clôture de J+1 = premier prix APRÈS l'information (prochaine séance de bourse)

    La variation est calculée sur les prix ajustés (adjClose), seuls comparables si une
    division d'actions ou un détachement de dividende tombe dans la fenêtre.
    """
    events = df_av[['fiscalDateEnding', 'reportedDate', 'reportedEPS', 'estimatedEPS',
                    'surprise', 'surprisePercentage']].dropna(subset=['reportedDate'])
    events = events.sort_values('reportedDate')

    prices = df_prices[['date', 'close', 'adjClose']].sort_values('date')
    before = prices.rename(columns={'date': 'date_J', 'close': 'close_J', 'adjClose': 'adjClose_J'})
    after = prices.rename(columns={'date': 'date_J1', 'close': 'close_J1', 'adjClose': 'adjClose_J1'})

    # Clôture de J : dernière séance <= J (= J lui-même si J est un jour de bourse)
    events = pd.merge_asof(events, before, left_on='reportedDate', right_on='date_J',
                           direction='backward', tolerance=pd.Timedelta('5 days'))
    # Clôture de J+1 : première séance strictement postérieure à J
    events = pd.merge_asof(events, after, left_on='reportedDate', right_on='date_J1',
                           direction='forward', allow_exact_matches=False,
                           tolerance=pd.Timedelta('5 days'))

    # Publications trop récentes (J+1 pas encore coté) ou hors de l'historique de prix
    events = events.dropna(subset=['close_J', 'close_J1']).copy()

    # Contrôle qualité : J devrait toujours être un jour de bourse
    events['J_jour_de_bourse'] = events['date_J'] == events['reportedDate']

    events['variation_pct'] = (events['adjClose_J1'] / events['adjClose_J'] - 1) * 100
    return events.reset_index(drop=True)


def regress_price_on_surprise(
    reaction: pd.DataFrame,
    x_col: str = 'surprisePercentage',
    y_col: str = 'variation_pct',
    start_date: str = '2010-01-01'
):
    """
    Régression linéaire par moindres carrés ordinaires (MCO), codée avec numpy seul :

        variation_pct_t = alpha + beta * surprisePercentage_t + e_t

    - Échantillon : publications à partir de start_date (les surprises des années 1996-2009
      sont extrêmes car le BPA d'Apple était proche de 0).
    - Écarts-types classiques ET robustes à l'hétéroscédasticité (White HC1).
      Les tests (t, p-value, intervalle à 95 %) utilisent les écarts-types robustes.
    - p-values approchées par la loi normale : avec ~70 observations,
      l'écart avec la loi de Student est négligeable (seuil 5 % : 2,00 au lieu de 1,96).

    Retourne (table des coefficients, dictionnaire de statistiques, échantillon utilisé).
    """
    sample = reaction[reaction['reportedDate'] >= start_date].dropna(subset=[x_col, y_col]).copy()
    x = sample[x_col].to_numpy(dtype=float)
    y = sample[y_col].to_numpy(dtype=float)
    n, k = len(y), 2
    if n <= k:
        raise ValueError(f"Pas assez d'observations ({n}) pour estimer la régression.")

    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta

    # Écarts-types classiques (homoscédasticité)
    rss = resid @ resid
    sigma2 = rss / (n - k)
    se_classic = np.sqrt(np.diag(sigma2 * XtX_inv))

    # Écarts-types robustes de White, correction HC1 : n / (n - k)
    meat = X.T @ (X * (resid ** 2)[:, None])
    cov_hc1 = XtX_inv @ meat @ XtX_inv * n / (n - k)
    se_robust = np.sqrt(np.diag(cov_hc1))

    t_stats = beta / se_robust
    p_values = [math.erfc(abs(t) / math.sqrt(2)) for t in t_stats]

    coef_table = pd.DataFrame({
        'coefficient': beta,
        'ecart_type_classique': se_classic,
        'ecart_type_robuste_HC1': se_robust,
        't_robuste': t_stats,
        'p_value': p_values,
        'IC95_bas': beta - 1.96 * se_robust,
        'IC95_haut': beta + 1.96 * se_robust,
    }, index=['constante (alpha)', f'{x_col} (beta)'])

    tss = ((y - y.mean()) ** 2).sum()
    stats = {
        'N_Observations': n,
        'Période': f"{sample['reportedDate'].min():%m/%Y} → {sample['reportedDate'].max():%m/%Y}",
        'R2': round(float(1 - rss / tss), 4),
        'R2_ajusté': round(float(1 - (rss / (n - k)) / (tss / (n - 1))), 4),
        'Corrélation': round(float(np.corrcoef(x, y)[0, 1]), 4),
        'Écart-type des résidus (pts de %)': round(math.sqrt(sigma2), 3),
    }
    return coef_table, stats, sample


# ==============================================================================
# 5. RAPPROCHEMENT ET ANALYSE COMPARATIVE (RÉEL VS PROJECTIONS)
# ==============================================================================

def compare_real_vs_projections(df_sec_eps: pd.DataFrame, df_av: pd.DataFrame) -> pd.DataFrame:
    """
    Rapproche les données réelles déclarées à la SEC avec les prévisions moyennes d'analystes.
    Utilise merge_asof pour aligner les fins de trimestres fiscales (tolérance 7 jours).
    """
    sec_cols = ['end', 'val', 'fy', 'fp', 'form'] + (['q4_derive'] if 'q4_derive' in df_sec_eps.columns else [])
    sec_clean = df_sec_eps[sec_cols].sort_values('end')
    sec_clean = sec_clean.rename(columns={'val': 'sec_real_eps', 'end': 'sec_date'})

    av_clean = df_av.sort_values('fiscalDateEnding').rename(columns={'fiscalDateEnding': 'av_date'})

    merged = pd.merge_asof(
        sec_clean,
        av_clean,
        left_on='sec_date',
        right_on='av_date',
        direction='nearest',
        tolerance=pd.Timedelta('7 days')
    )

    merged = merged.dropna(subset=['estimatedEPS']).copy()
    
    # Calcul des erreurs de prévision : e_t = Réel - Projection
    merged['forecast_error'] = merged['sec_real_eps'] - merged['estimatedEPS']
    merged['forecast_pct_error'] = (merged['forecast_error'] / merged['estimatedEPS'].abs()) * 100
    merged['beat_projection'] = merged['forecast_error'] > 0
    
    return merged

def compute_forecast_metrics(df_comparison: pd.DataFrame) -> dict:
    """Calcule les indicateurs statistiques de précision des prévisions."""
    errors = df_comparison['forecast_error']
    pct_errors = df_comparison['forecast_pct_error']
    
    mae = errors.abs().mean()
    rmse = np.sqrt((errors ** 2).mean())
    bias = errors.mean()  # Biais moyen (positif = consensus trop pessimiste)
    mape = pct_errors.abs().mean()
    beat_rate = df_comparison['beat_projection'].mean() * 100
    
    return {
        'N_Observations': len(df_comparison),
        'MAE (USD/action)': round(float(mae), 4),
        'RMSE (USD/action)': round(float(rmse), 4),
        'Biais Moyen (USD/action)': round(float(bias), 4),
        'MAPE (%)': round(float(mape), 2),
        'Taux Surprise Positive (%)': round(float(beat_rate), 2)
    }


# ==============================================================================
# 6. VISUALISATION GRAPHIQUE
# ==============================================================================

def filter_from_report_date(df_comparison: pd.DataFrame, start_date: str = None) -> pd.DataFrame:
    """
    Garde les trimestres publiés à partir de start_date (colonne 'reportedDate' d'Alpha Vantage),
    soit le même critère que regress_price_on_surprise : les graphiques couvrent ainsi
    exactement les publications de l'échantillon de régression.
    """
    df = df_comparison.sort_values('sec_date').copy()
    if start_date is not None:
        df = df[df['reportedDate'] >= start_date]
    return df


def plot_real_vs_projected(df_comparison: pd.DataFrame, start_date: str = None, save_path: str = None):
    """
    Graphique Réel vs Consensus d'analystes, un point par trimestre.
    df_comparison doit venir de compare_real_vs_projections appliqué à build_quarterly_eps
    (BPA ajusté des divisions d'actions, T4 reconstitués) : tous les points sont trimestriels.
    """
    sns.set_theme(style="whitegrid")
    df = filter_from_report_date(df_comparison, start_date)

    plt.figure(figsize=(12, 6))
    plt.plot(df['sec_date'], df['sec_real_eps'], marker='o', color='#1f77b4', linewidth=2, label='BPA Réel (SEC EDGAR)')
    plt.plot(df['sec_date'], df['estimatedEPS'], marker='s', linestyle='--', color='#ff7f0e', linewidth=2, label='Consensus Prévisionnel (Alpha Vantage)')

    plt.fill_between(
        df['sec_date'],
        df['sec_real_eps'],
        df['estimatedEPS'],
        where=(df['sec_real_eps'] >= df['estimatedEPS']),
        interpolate=True,
        color='green', alpha=0.2, label='Surprise Positive (Réel > Projection)'
    )
    plt.fill_between(
        df['sec_date'],
        df['sec_real_eps'],
        df['estimatedEPS'],
        where=(df['sec_real_eps'] < df['estimatedEPS']),
        interpolate=True,
        color='red', alpha=0.2, label='Surprise Négative (Réel < Projection)'
    )

    plt.title(f"Apple Inc. (AAPL) : BPA trimestriel Réel vs Consensus Prévisionnel "
              f"(publications {df['reportedDate'].min():%Y}-{df['reportedDate'].max():%Y}, {len(df)} trimestres)",
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Date de clôture du trimestre", fontsize=11)
    plt.ylabel("BPA Dilué (USD / action)", fontsize=11)
    plt.legend(frameon=True, loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_price_vs_surprise(sample: pd.DataFrame, coef_table: pd.DataFrame,
                           x_col: str = 'surprisePercentage', y_col: str = 'variation_pct',
                           save_path: str = None):
    """Nuage de points variation du cours (J -> J+1) vs surprise de BPA, avec la droite MCO."""
    sns.set_theme(style="whitegrid")
    alpha, beta = coef_table['coefficient'].iloc[0], coef_table['coefficient'].iloc[1]
    p_beta = coef_table['p_value'].iloc[1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color='#c9c8c3', linewidth=1, zorder=1)
    ax.axvline(0, color='#c9c8c3', linewidth=1, zorder=1)

    ax.scatter(sample[x_col], sample[y_col], s=55, color='#2a78d6', alpha=0.85,
               edgecolor='white', linewidth=1.5, zorder=3, label='Publication de résultats')

    x_line = np.linspace(sample[x_col].min(), sample[x_col].max(), 100)
    ax.plot(x_line, alpha + beta * x_line, color='#52514e', linewidth=2, zorder=2,
            label=f'Droite MCO : {alpha:.2f} + {beta:.3f} × surprise  (p = {p_beta:.3f})')

    ax.set_title("Apple (AAPL) : réaction du cours à la surprise de BPA (J → J+1)",
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Surprise de BPA (% du consensus)", fontsize=11)
    ax.set_ylabel("Variation du cours J → J+1 (%)", fontsize=11)
    ax.legend(frameon=True, loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_error_distribution(df_comparison: pd.DataFrame, start_date: str = None, save_path: str = None):
    """Distribution des erreurs de prévision trimestrielles (Réel - Projection)."""
    sns.set_theme(style="whitegrid")
    df = filter_from_report_date(df_comparison, start_date)

    plt.figure(figsize=(9, 5))
    sns.histplot(df['forecast_error'], kde=True, color='#2ca02c', bins=20)
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Erreur nulle (Projection exacte)')
    plt.axvline(df['forecast_error'].mean(), color='blue', linestyle=':', linewidth=2,
                label=f'Biais moyen = {df["forecast_error"].mean():+.2f} $')

    plt.title(f"Distribution des erreurs de prévision trimestrielles (Réel - Projection), "
              f"publications {df['reportedDate'].min():%Y}-{df['reportedDate'].max():%Y}", fontsize=13, fontweight='bold')
    plt.xlabel("Écart de BPA (USD / action)", fontsize=11)
    plt.ylabel("Fréquence (Trimestres)", fontsize=11)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
