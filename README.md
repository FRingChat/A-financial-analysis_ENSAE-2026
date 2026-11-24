# Prévisions des analystes et réaction du marché : le cas d'Apple

Projet ENSAE (2e année) d'exploration de données avec Pandas.

Nous mesurons la précision du consensus des analystes sur le bénéfice par action (BPA) trimestriel d'Apple, puis l'effet de la surprise (écart entre le BPA réel et le consensus) sur le cours de bourse le lendemain de la publication. La période étudiée couvre les publications de janvier 2010 à novembre 2026.

## Sources de données

| Source | Contenu | Accès |
|---|---|---|
| SEC EDGAR (*Company Facts*) | BPA dilué officiel (10-Q / 10-K) | API publique, sans clé |
| Alpha Vantage (`EARNINGS`) | Consensus des analystes, BPA publié, surprise | Clé d'API, via notre proxy Vercel |
| Tiingo | Cours quotidiens (bruts et ajustés), divisions d'actions | Clé d'API, via notre proxy Vercel |

## Organisation du dépôt

- `Main.ipynb` : le notebook d'analyse (à lire en premier).
- `fonctions.py` : récupération des données, nettoyage, jointures, régression et graphiques.
- `app.py` (+ `api/index.py`, `vercel.json`) : proxy déployé sur Vercel, qui détient les clés d'API Alpha Vantage et Tiingo. Il limite chaque utilisateur à 5 requêtes par jour et par route (limite indicative, gardée en mémoire par le serveur).
- `data_cache/` (non versionné) : copie locale des réponses des API, créée à la première exécution.

## Lancer le notebook

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate    |    macOS / Linux : source .venv/bin/activate
pip install -r requirements.txt
```

Ouvrez ensuite `Main.ipynb` et exécutez toutes les cellules.

Aucune clé d'API n'est nécessaire : la première exécution télécharge les données (SEC en direct, Alpha Vantage et Tiingo via le proxy Vercel) et les enregistre dans `data_cache/`. Les exécutions suivantes n'utilisent plus le réseau.

Si le proxy n'est pas disponible, les fonctions utilisent une clé locale : créez un fichier `.env.local` (ignoré par git) contenant `ALPHAVANTAGE_API_KEY=...` et `TIINGO_API_KEY=...`, ou récupérez-le depuis Vercel avec `npx vercel link` puis `npx vercel env pull .env.local`.
