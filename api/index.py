"""
API Serverless Vercel : Proxy sécurisé pour Alpha Vantage
- Protège la clé secrète ALPHAVANTAGE_API_KEY sur les serveurs de Vercel.
- Limite stricte : 5 requêtes par utilisateur (adresse IP) par tranche de 24h.
- Cache serveur : Conserve la réponse Alpha Vantage pendant 24h pour ne pas
  consommer inutilement le forfait Alpha Vantage.
"""

from http.server import BaseHTTPRequestHandler
import urllib.request
import urllib.parse
import json
import time
import os

# Mémoire partagée (warm instances Vercel)
IP_REQUESTS = {}  # { ip: [timestamp1, timestamp2, ...] }
DATA_CACHE = {}   # { symbol: { "data": dict, "timestamp": float } }

DAILY_LIMIT = 5
WINDOW_SECONDS = 86400  # 24 heures glissantes

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed_url.query)
        symbol = params.get('symbol', ['AAPL'])[0].upper()

        # 1. Identification de l'utilisateur par son adresse IP
        client_ip = (
            self.headers.get('x-real-ip')
            or self.headers.get('x-forwarded-for', '127.0.0.1').split(',')[0].strip()
        )

        now = time.time()

        # 2. Contrôle du Rate Limiting (5 requêtes / jour / IP)
        timestamps = IP_REQUESTS.get(client_ip, [])
        valid_timestamps = [t for t in timestamps if now - t < WINDOW_SECONDS]

        if len(valid_timestamps) >= DAILY_LIMIT:
            self.send_response(429)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('X-RateLimit-Limit', str(DAILY_LIMIT))
            self.send_header('X-RateLimit-Remaining', '0')
            self.end_headers()

            error_body = {
                "error": "Quota dépassé",
                "message": f"Vous avez atteint la limite de {DAILY_LIMIT} requêtes par jour par utilisateur.",
                "remaining": 0,
                "reset_in_seconds": int(WINDOW_SECONDS - (now - valid_timestamps[0]))
            }
            self.wfile.write(json.dumps(error_body, ensure_ascii=False).encode('utf-8'))
            return

        # 3. Vérification du cache serveur Vercel
        cached = DATA_CACHE.get(symbol)
        if cached and (now - cached['timestamp'] < WINDOW_SECONDS):
            response_data = cached['data']
        else:
            api_key = os.environ.get('ALPHAVANTAGE_API_KEY')
            if not api_key:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                err = {
                    "error": "Configuration serveur manquante",
                    "message": "La variable ALPHAVANTAGE_API_KEY n'est pas configurée dans les paramètres Vercel."
                }
                self.wfile.write(json.dumps(err, ensure_ascii=False).encode('utf-8'))
                return

            av_url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={api_key}"
            try:
                req = urllib.request.Request(av_url, headers={'User-Agent': 'Vercel-Proxy/1.0'})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    response_data = json.loads(resp.read().decode('utf-8'))

                if "quarterlyEarnings" in response_data:
                    DATA_CACHE[symbol] = {
                        "data": response_data,
                        "timestamp": now
                    }
            except Exception as e:
                self.send_response(502)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                err = {"error": "Erreur lors de l'appel à Alpha Vantage", "details": str(e)}
                self.wfile.write(json.dumps(err, ensure_ascii=False).encode('utf-8'))
                return

        # Mise à jour du compteur pour cette IP
        valid_timestamps.append(now)
        IP_REQUESTS[client_ip] = valid_timestamps
        remaining = max(0, DAILY_LIMIT - len(valid_timestamps))

        # 4. Envoi de la réponse JSON avec en-têtes de cache Vercel Edge
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'public, s-maxage=43200, stale-while-revalidate=21600')
        self.send_header('X-RateLimit-Limit', str(DAILY_LIMIT))
        self.send_header('X-RateLimit-Remaining', str(remaining))
        self.end_headers()

        self.wfile.write(json.dumps(response_data).encode('utf-8'))
