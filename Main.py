import os
import json
import urllib.request
import pandas as pd
import numpy as np

# ==============================================================================
# 1. GESTION SÉCURISÉE DE LA CLÉ D'API ALPHAVANTAGE (SÉCURITÉ)
# ==============================================================================
def load_env(env_path: str = ".env") -> None:
    """Charge les variables d'environnement depuis un fichier local .env sans dépendance externe."""
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")

def get_alphavantage_key() -> str:
    """
    Récupère la clé secrète de manière sécurisée.
    Ne JAMAIS afficher ni inscrire la clé en clair dans le code source.
    """
    load_env()
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        raise ValueError(
            "\n[ERREUR SÉCURITÉ] Clé Alpha Vantage absente ou non configurée.\n"
            "Pour protéger votre clé :\n"
            "1. Créez un fichier '.env' à la racine du projet (déjà ignoré par git).\n"
            "2. Ajoutez-y la ligne : ALPHAVANTAGE_API_KEY=votre_cle_secrete\n"
        )
    return key


# ==============================================================================
# 2. EXTRACTION DES DONNÉES RÉELLES OFFICIELLES (SEC EDGAR)
# ==============================================================================
def fetch_sec_data(cik: str = "CIK0000320193", cache_file: str = "data_cache/sec_apple.json") -> dict:
    """Récupère et met en cache les données officielles SEC EDGAR pour Apple."""
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

def extract_sec_metric(facts: dict, taxonomy: str, metric_name: str, quarterly_only: bool = True) -> pd.DataFrame:
    """Extrait une métrique sous forme de DataFrame avec conversion temporelle."""
    if metric_name not in facts.get(taxonomy, {}):
        raise KeyError(f"Métrique '{metric_name}' non trouvée dans {taxonomy}")
        
    units_dict = facts[taxonomy][metric_name]['units']
    unit_key = list(units_dict.keys())[0]
    df = pd.DataFrame(units_dict[unit_key])
    
    df['end'] = pd.to_datetime(df['end'])
    df['filed'] = pd.to_datetime(df['filed'])
    
    if quarterly_only:
        # On garde les déclarations trimestrielles (10-Q) et annuelles (10-K pour le T4)
        df = df[df['form'].isin(['10-Q', '10-K'])].copy()
        if 'start' in df.columns:
            df['start'] = pd.to_datetime(df['start'])
            # Filtre sur les périodes d'environ 3 mois (pour éliminer les cumuls semestriels/9 mois)
            df['duration_days'] = (df['end'] - df['start']).dt.days
            df = df[(df['duration_days'] >= 75) & (df['duration_days'] <= 105)].copy()
            
        # Déduplication en cas de rectificatif
        df = df.sort_values('filed').groupby('end').last().reset_index()
        
    df['metric_name'] = metric_name
    return df


# ==============================================================================
# 3. EXTRACTION DES PROJECTIONS & ESTIMATIONS (ALPHA VANTAGE)
# ==============================================================================
def fetch_alphavantage_earnings(symbol: str = "AAPL", cache_dir: str = "data_cache") -> pd.DataFrame:
    """
    Récupère l'historique des résultats réels vs prévisions (consensus des analystes).
    Utilise un cache local pour respecter le quota gratuit (25 requêtes/jour).
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

    # Conversion en DataFrame
    df = pd.DataFrame(raw_data.get("quarterlyEarnings", []))
    if df.empty:
        return df
        
    df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"])
    df["reportedDate"] = pd.to_datetime(df["reportedDate"])
    
    # Typage numérique des métriques de projection
    numeric_cols = ["reportedEPS", "estimatedEPS", "surprise", "surprisePercentage"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    return df


# ==============================================================================
# 4. COMPARAISON RIGOUREUSE : RÉEL (SEC) VS PROJECTION (ALPHA VANTAGE)
# ==============================================================================
def compare_real_vs_projections(df_sec_eps: pd.DataFrame, df_av: pd.DataFrame) -> pd.DataFrame:
    """
    Rapproche les données réelles déclarées à la SEC avec les estimations d'analystes.
    Utilise merge_asof pour aligner les fins de trimestres fiscales (à +/- 7 jours).
    """
    sec_clean = df_sec_eps[['end', 'val', 'fy', 'fp', 'form']].sort_values('end')
    sec_clean = sec_clean.rename(columns={'val': 'sec_real_eps', 'end': 'sec_date'})
    
    av_clean = df_av.sort_values('fiscalDateEnding').rename(columns={'fiscalDateEnding': 'av_date'})
    
    # Rapprochement chronologique avec tolérance de 7 jours
    merged = pd.merge_asof(
        sec_clean,
        av_clean,
        left_on='sec_date',
        right_on='av_date',
        direction='nearest',
        tolerance=pd.Timedelta('7 days')
    )
    
    # Calcul des écarts de prévision
    merged['forecast_error'] = merged['sec_real_eps'] - merged['estimatedEPS']
    merged['forecast_pct_error'] = (merged['forecast_error'] / merged['estimatedEPS'].abs()) * 100
    merged['beat_projection'] = merged['forecast_error'] > 0
    
    return merged.dropna(subset=['estimatedEPS'])


# ==============================================================================
# EXÉCUTION
# ==============================================================================
if __name__ == "__main__":
    print("--- 1. Extraction des données SEC réelles ---")
    sec_raw = fetch_sec_data()
    df_sec_eps = extract_sec_metric(sec_raw['facts'], 'us-gaap', 'EarningsPerShareDiluted', quarterly_only=True)
    print(f"BPA trimestriel SEC extrait : {len(df_sec_eps)} trimestres disponibles.")

    print("\n--- 2. Extraction des données Alpha Vantage ---")
    try:
        df_av = fetch_alphavantage_earnings("AAPL")
        print(f"Projections Alpha Vantage extraites : {len(df_av)} trimestres.")
        
        print("\n--- 3. Comparaison Réel vs Projections ---")
        comparison = compare_real_vs_projections(df_sec_eps, df_av)
        
        # Statistiques récapitulatives
        mae = comparison['forecast_error'].abs().mean()
        beat_rate = comparison['beat_projection'].mean() * 100
        
        print(f"Nombre de trimestres appariés : {len(comparison)}")
        print(f"Erreur absolue moyenne (MAE) : {mae:.4f} USD par action")
        print(f"Taux de surprise positive (Battu le consensus) : {beat_rate:.1f} %")
        print("\nDernières observations comparées :")
        cols_show = ['sec_date', 'sec_real_eps', 'estimatedEPS', 'forecast_error', 'forecast_pct_error']
        print(comparison[cols_show].tail(5).to_string(index=False))
        
    except ValueError as e:
        print(e)
