import json
import urllib.request
import pandas as pd


# Récupération des données SEC avec un User-Agent conforme
url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
req = urllib.request.Request(url, headers={'User-Agent': 'ENSAE-Student contact@ensae.fr'})

with urllib.request.urlopen(req) as response:
    company_facts = json.loads(response.read().decode('utf-8'))

facts = company_facts['facts']

# Fonction utilitaire pour extraire une métrique en DataFrame propre
def get_metric_df(taxonomy: str, metric_name: str, annual_only: bool = True) -> pd.DataFrame:
    if metric_name not in facts[taxonomy]:
        raise KeyError(f"Métrique {metric_name} introuvable dans {taxonomy}")
    
    metric_data = facts[taxonomy][metric_name]
    units_dict = metric_data['units']
    unit_key = list(units_dict.keys())[0]  # 'USD', 'shares', 'USD/shares'
    
    df = pd.DataFrame(units_dict[unit_key])
    df['end'] = pd.to_datetime(df['end'])
    df['filed'] = pd.to_datetime(df['filed'])
    
    if annual_only:
        # On garde uniquement les rapports annuels 10-K pour des exercices complets
        df = df[(df['form'] == '10-K') & (df['fp'] == 'FY')].copy()
        # En cas de redressement (10-K/A), on garde la dernière version déposée
        df = df.sort_values('filed').groupby('fy').last().reset_index()
    
    df['metric'] = metric_name
    return df


# Chiffre d'affaires annuel
df_revenue = get_metric_df('us-gaap', 'RevenueFromContractWithCustomerExcludingAssessedTax')
# Résultat net annuel
df_net_income = get_metric_df('us-gaap', 'NetIncomeLoss')
# Nombre d'actions (DEI)
df_shares = get_metric_df('dei', 'EntityCommonStockSharesOutstanding', annual_only=True)