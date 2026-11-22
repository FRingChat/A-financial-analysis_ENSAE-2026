"""
Module fonctions.py
Contient l'ensemble des fonctions utilitaires pour :
- Récupérer la clé d'API Alpha Vantage stockée de manière sécurisée sur Vercel (.env.local ou os.environ).
- Récupérer et mettre en cache les données comptables SEC EDGAR (Apple CIK 0000320193).
- Récupérer et mettre en cache les projections/consensus des analystes via Alpha Vantage.
- Aligner temporellement les séries temporelles et calculer les métriques d'erreur de prévision.
- Visualiser graphiquement les résultats réels vs prévisions.
"""

import os
import json
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

def get_alphavantage_key() -> str:
    """
    Récupère la clé Alpha Vantage stockée sur Vercel :
    1. Directement depuis os.environ (environnement Vercel en production / cloud)
    2. Depuis .env.local (fichier généré localement par 'npx vercel env pull')
    3. Depuis .env (fichier local de secours)
    4. Tente une synchronisation automatique si possible
    """
    # 1. Vérification dans l'environnement système
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if key and key != "VOTRE_CLE_ICI":
        return key

    # 2. Chargement des fichiers d'environnement locaux (.env.local généré par Vercel)
    load_vercel_env()
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if key and key != "VOTRE_CLE_ICI":
        return key

    # 3. Tentative de synchronisation via Vercel CLI si le projet est lié
    print("Tentative de synchronisation des variables d'environnement depuis Vercel...")
    if pull_vercel_env():
        key = os.environ.get("ALPHAVANTAGE_API_KEY")
        if key and key != "VOTRE_CLE_ICI":
            print("Variables d'environnement synchronisées depuis Vercel avec succès !")
            return key

    # 4. Message explicite d'aide à la synchronisation Vercel
    raise ValueError(
        "\n========================================================================\n"
        "🔑 CLÉ ALPHAVANTAGE INTROUVABLE EN LOCAL\n"
        "Votre clé est configurée dans votre projet Vercel, mais n'est pas encore\n"
        "synchronisée sur votre machine.\n\n"
        "Pour synchroniser votre clé depuis Vercel en 2 étapes rapides :\n"
        "1. Dans votre terminal, liez votre dossier au projet Vercel :\n"
        "   npx vercel link\n"
        "2. Téléchargez votre variable d'environnement :\n"
        "   npx vercel env pull .env.local\n\n"
        "(Le fichier .env.local est déjà protégé dans .gitignore et ne sera jamais publié)\n"
        "========================================================================"
    )


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
    vercel_url: str = "https://projetprog-financeaugustinfelix-ensae-prog-27.vercel.app"
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
# 4. RAPPROCHEMENT ET ANALYSE COMPARATIVE (RÉEL VS PROJECTIONS)
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
# 5. VISUALISATION GRAPHIQUE
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
