import pandas as pd

# Import the first dataset with some credit cards
url = "https://cdn.jsdelivr.net/gh/thedavidweng/opencard-db@v0.2.0/exports/cards.csv"
df_cards = pd.read_csv(url)

# We count the number of inactives credit cards
df_inactifs = df_cards[df_cards['status'] != 'active']
if df_inactifs.shape[0] == 0:
    print("Toutes les cartes sont actives")
else:
    print(df_inactifs.shape)


# Import a second dataset with other informations about those credit cards
url = "https://cdn.jsdelivr.net/gh/thedavidweng/opencard-db@v0.2.0/exports/cards-all.json"
df_cards_all = pd.read_json(url)
