"""
Module fonctions.py
Contient l'ensemble des fonctions utilitaires pour :
- Charger la configuration et sécuriser la clé d'API Alpha Vantage.
- Récupérer et mettre en cache les données comptables SEC EDGAR (Apple CIK 0000320193).
- Récupérer et mettre en cache les projections/consensus des analystes via Alpha Vantage.
- Aligner temporellement les séries temporelles et calculer les métriques d'erreur de prévision.
- Visualiser graphiquement les résultats réels vs prévisions.
"""

import os
import json
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================================
# 1. GESTION SÉCURISÉE DE LA CLÉ D'API
# ==============================================================================

def load_env(env_path: str = ".env") -> None:
    """Charge les variables d'environnement depuis le fichier .env sans dépendance externe."""
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")

def get_alphavantage_key() -> str:
    """
    Récupère la clé secrète Alpha Vantage depuis l'environnement ou .env.
    Lève une exception explicite si la clé est absente pour protéger le projet.
    """
    load_env()
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key or key == "VOTRE_CLE_ICI":
        raise ValueError(
            "\n[ERREUR SÉCURITÉ] Clé Alpha Vantage absente ou non configurée.\n"
            "Pour protéger votre clé :\n"
            "1. Créez un fichier '.env' à la racine du projet.\n"
            "2. Ajoutez-y : ALPHAVANTAGE_API_KEY=votre_cle_secrete\n"
        )
    return key


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

def fetch_alphavantage_earnings(symbol: str = "AAPL", cache_dir: str = "data_cache") -> pd.DataFrame:
    """
    Récupère l'historique des consensus de projections et résultats réels via Alpha Vantage.
    Utilise un cache local pour respecter le quota gratuit (25 requêtes / jour).
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"av_earnings_{symbol}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        api_key = get_alphavantage_key()
        url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={api_key}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            raw_data = json.loads(resp.read().decode('utf-8'))

        if "Error Message" in raw_data:
            raise ValueError(f"Erreur Alpha Vantage : {raw_data['Error Message']}")
        if "Information" in raw_data:
            raise ValueError(f"Note Alpha Vantage : {raw_data['Information']}")

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2)

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
