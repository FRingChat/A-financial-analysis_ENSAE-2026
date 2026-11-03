import pandas as pd

# We import the dataset using a special key to outpass anti-bot systems
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
}
url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
df_global = pd.read_json(url, storage_options=headers)


# The last column ('facts') is a JSON considered as a text
# We first explore the first line (apple dei)
# It itself contained two dictionary, one common and one public
dic_temp = (df_global.facts.iloc[0])['EntityCommonStockSharesOutstanding']
df_common = pd.DataFrame([dic_temp])
dic_temp = (df_global.facts.iloc[0])['EntityPublicFloat']
df_public = pd.DataFrame([dic_temp])

print(df_common.describe)
