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
        cmd = f"npx vercel env pull {target_file} --yes"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(target_file):
            load_vercel_env((target_file,))
            return True
    except Exception:
        pass
    return False

def get_api_key(var_name: str) -> str:
    """
    Récupère une clé d'API stockée sur Vercel (ex: 'ALPHAVANTAGE_API_KEY', 'TIINGO_API_KEY') :
    1. Directement depuis os.environ (environnement Vercel en production / cloud)
    2. Depuis .env.local (fichier généré localement par 'npx vercel env pull')
    3. Depuis .env (fichier local de secours)
    4. Tente une synchronisation automatique si possible
    """
    # 1. Vérification dans l'environnement système
    key = os.environ.get(var_name)
    if key and key != "VOTRE_CLE_ICI":
        return key

    # 2. Chargement des fichiers d'environnement locaux (.env.local généré par Vercel)
    load_vercel_env()
    key = os.environ.get(var_name)
    if key and key != "VOTRE_CLE_ICI":
        return key

    # 3. Tentative de synchronisation via Vercel CLI si le projet est lié
    print("Tentative de synchronisation des variables d'environnement depuis Vercel...")
    if pull_vercel_env():
        key = os.environ.get(var_name)
        if key and key != "VOTRE_CLE_ICI":
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
    """Récupère la clé Tiingo (variable TIINGO_API_KEY)."""
    return get_api_key("TIINGO_API_KEY")


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
    """
    if metric_name not in facts.get(taxonomy, {}):
        raise KeyError(f"Métrique '{metric_name}' non trouvée dans la taxonomie {taxonomy}")

    units_dict = facts[taxonomy][metric_name]['units']
    unit_key = list(units_dict.keys())[0]
    df = pd.DataFrame(units_dict[unit_key])

    df['end'] = pd.to_datetime(df['end'])
    df['filed'] = pd.to_datetime(df['filed'])

    if frequency == 'quarterly':
        df = df[df['form'].isin(['10-Q', '10-K'])].copy()
        if 'start' in df.columns:
            df['start'] = pd.to_datetime(df['start'])
            # Périodes d'environ 3 mois (75 à 105 jours) pour éliminer les cumuls semestriels ou annuels
            df['duration_days'] = (df['end'] - df['start']).dt.days
            df = df[(df['duration_days'] >= 75) & (df['duration_days'] <= 105)].copy()
        # Déduplication en cas de dépôt modificatif (garde le dépôt le plus récent)
        df = df.sort_values('filed').groupby('end').last().reset_index()

    elif frequency == 'annual':
        df = df[(df['form'] == '10-K') & (df['fp'] == 'FY')].copy()
        df = df.sort_values('filed').groupby('fy').last().reset_index()

    df['metric_name'] = metric_name
    return df


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
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        raw_data = None
        proxy_error = None

        # 2. Tentative via le Proxy Vercel (sans clé pour l'utilisateur)
        proxy_endpoint = f"{vercel_url.rstrip('/')}/api/earnings?symbol={symbol}"
        try:
            print(f"Tentative de récupération via le proxy Vercel ({proxy_endpoint})...")
            req = urllib.request.Request(proxy_endpoint, headers={'User-Agent': 'ENSAE-Pandas-Client/1.0'})
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw_data = json.loads(resp.read().decode('utf-8'))

            if isinstance(raw_data, dict) and "error" in raw_data and raw_data.get("error") == "Quota dépassé":
                raise PermissionError(f"[Quota Vercel] {raw_data.get('message')}")

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, indent=2)
            print("Données récupérées avec succès via l'API Vercel !")

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
        if not raw_data or "quarterlyEarnings" not in raw_data:
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

    - Échantillon : publications à partir de start_date (les surprises des années 1996-2002
      sont extrêmes car le BPA d'Apple était proche de 0).
    - Écarts-types classiques ET robustes à l'hétéroscédasticité (White HC1).
      Les tests (t, p-value, intervalle à 95 %) utilisent les écarts-types robustes.
    - p-values approchées par la loi normale : avec ~65 observations,
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
    sec_clean = df_sec_eps[['end', 'val', 'fy', 'fp', 'form']].sort_values('end')
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
        'MAE (USD/action)': round(mae, 4),
        'RMSE (USD/action)': round(rmse, 4),
        'Biais Moyen (USD/action)': round(bias, 4),
        'MAPE (%)': round(mape, 2),
        'Taux Surprise Positive (%)': round(beat_rate, 2)
    }


# ==============================================================================
# 6. VISUALISATION GRAPHIQUE
# ==============================================================================

def plot_real_vs_projected(df_comparison: pd.DataFrame, recent_only: bool = True, save_path: str = None):
    """Génère un graphique comparatif Réel vs Consensus d'analystes."""
    sns.set_theme(style="whitegrid")
    df = df_comparison.copy()
    
    # Apple a effectué un split d'actions 4:1 en août 2020. Filtrer sur les années récentes (post-2020)
    # permet d'avoir des ordres de grandeur parfaitement comparables pour la visualisation
    if recent_only:
        df = df[df['sec_date'] >= '2020-10-01']

    plt.figure(figsize=(12, 6))
    plt.plot(df['sec_date'], df['sec_real_eps'], marker='o', color='#1f77b4', linewidth=2, label='BPA Réel (SEC EDGAR)')
    plt.plot(df['sec_date'], df['estimatedEPS'], marker='s', linestyle='--', color='#ff7f0e', linewidth=2, label='Consensus Prévisionnel (Alpha Vantage)')

    plt.fill_between(
        df['sec_date'],
        df['sec_real_eps'],
        df['estimatedEPS'],
        where=(df['sec_real_eps'] >= df['estimatedEPS']),
        color='green', alpha=0.2, label='Surprise Positive (Réel > Projection)'
    )
    plt.fill_between(
        df['sec_date'],
        df['sec_real_eps'],
        df['estimatedEPS'],
        where=(df['sec_real_eps'] < df['estimatedEPS']),
        color='red', alpha=0.2, label='Surprise Négative (Réel < Projection)'
    )

    plt.title("Apple Inc. (AAPL) : Bénéfice par Action Réel vs Consensus Prévisionnel", fontsize=14, fontweight='bold', pad=15)
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

def plot_error_distribution(df_comparison: pd.DataFrame, recent_only: bool = True, save_path: str = None):
    """Génère la distribution des erreurs de prévision (Surprises)."""
    sns.set_theme(style="whitegrid")
    df = df_comparison.copy()
    if recent_only:
        df = df[df['sec_date'] >= '2020-10-01']

    plt.figure(figsize=(9, 5))
    sns.histplot(df['forecast_error'], kde=True, color='#2ca02c', bins=12)
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Erreur nulle (Projection exacte)')
    plt.axvline(df['forecast_error'].mean(), color='blue', linestyle=':', linewidth=2, 
                label=f'Biais moyen = +{df["forecast_error"].mean():.2f} $')

    plt.title("Distribution des erreurs de prévision (Réel - Projection)", fontsize=13, fontweight='bold')
    plt.xlabel("Écart de BPA (USD / action)", fontsize=11)
    plt.ylabel("Fréquence (Trimestres)", fontsize=11)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
