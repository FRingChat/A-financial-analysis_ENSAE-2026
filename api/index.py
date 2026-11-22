"""
API Serverless Vercel : Proxy sécurisé pour Alpha Vantage
Compatible à la fois avec le runtime WSGI/ASGI moderne de Vercel (app)
et le runtime standard Serverless (handler).
"""

from http.server import BaseHTTPRequestHandler
import urllib.request
import urllib.parse
import json
import time
import os

IP_REQUESTS = {}  # { ip: [timestamp1, timestamp2, ...] }
DATA_CACHE = {}   # { symbol: { "data": dict, "timestamp": float } }

DAILY_LIMIT = 5
WINDOW_SECONDS = 86400  # 24 heures glissantes

def process_earnings_request(symbol: str, client_ip: str):
    """Logique commune de traitement avec rate limit et cache."""
    now = time.time()
    timestamps = IP_REQUESTS.get(client_ip, [])
    valid_timestamps = [t for t in timestamps if now - t < WINDOW_SECONDS]

    # Vérification du Rate Limiting
    if len(valid_timestamps) >= DAILY_LIMIT:
        error_body = {
            "error": "Quota dépassé",
            "message": f"Vous avez atteint la limite de {DAILY_LIMIT} requêtes par jour par utilisateur.",
            "remaining": 0,
            "reset_in_seconds": int(WINDOW_SECONDS - (now - valid_timestamps[0]))
        }
        return 429, error_body, {'X-RateLimit-Limit': str(DAILY_LIMIT), 'X-RateLimit-Remaining': '0'}

    # Vérification du cache serveur Vercel
    cached = DATA_CACHE.get(symbol)
    if cached and (now - cached['timestamp'] < WINDOW_SECONDS):
        response_data = cached['data']
    else:
        api_key = os.environ.get('ALPHAVANTAGE_API_KEY')
        if not api_key:
            err = {
                "error": "Configuration serveur manquante",
                "message": "La variable ALPHAVANTAGE_API_KEY n'est pas configurée dans les paramètres Vercel."
            }
            return 500, err, {}

        av_url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={api_key}"
        try:
            req = urllib.request.Request(av_url, headers={'User-Agent': 'Vercel-Proxy/1.0'})
            with urllib.request.urlopen(req, timeout=12) as resp:
                response_data = json.loads(resp.read().decode('utf-8'))

            if "quarterlyEarnings" in response_data:
                DATA_CACHE[symbol] = {
                    "data": response_data,
                    "timestamp": now
                }
        except Exception as e:
            err = {"error": "Erreur lors de l'appel à Alpha Vantage", "details": str(e)}
            return 502, err, {}

    valid_timestamps.append(now)
    IP_REQUESTS[client_ip] = valid_timestamps
    remaining = max(0, DAILY_LIMIT - len(valid_timestamps))

    headers = {
        'Cache-Control': 'public, s-maxage=43200, stale-while-revalidate=21600',
        'X-RateLimit-Limit': str(DAILY_LIMIT),
        'X-RateLimit-Remaining': str(remaining)
    }
    return 200, response_data, headers


# 1. Point d'entrée WSGI moderne (attendu par le nouveau runtime Python Vercel)
def app(environ, start_response):
    method = environ.get('REQUEST_METHOD', 'GET')
    if method == 'OPTIONS':
        start_response('200 OK', [
            ('Access-Control-Allow-Origin', '*'),
            ('Access-Control-Allow-Methods', 'GET, OPTIONS'),
            ('Access-Control-Allow-Headers', 'Content-Type')
        ])
        return [b'']

    query_string = environ.get('QUERY_STRING', '')
    params = urllib.parse.parse_qs(query_string)
    symbol = params.get('symbol', ['AAPL'])[0].upper()

    client_ip = (
        environ.get('HTTP_X_REAL_IP')
        or environ.get('HTTP_X_FORWARDED_FOR', '127.0.0.1').split(',')[0].strip()
    )

    status_code, body_data, extra_headers = process_earnings_request(symbol, client_ip)
    
    status_text = {
        200: '200 OK',
        429: '429 Too Many Requests',
        500: '500 Internal Server Error',
        502: '502 Bad Gateway'
    }.get(status_code, '200 OK')

    resp_headers = [
        ('Content-Type', 'application/json; charset=utf-8'),
        ('Access-Control-Allow-Origin', '*')
    ]
    for k, v in extra_headers.items():
        resp_headers.append((k, v))

    start_response(status_text, resp_headers)
    return [json.dumps(body_data).encode('utf-8')]


# 2. Point d'entrée BaseHTTPRequestHandler classique (compatibilité Vercel functions)
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

        client_ip = (
            self.headers.get('x-real-ip')
            or self.headers.get('x-forwarded-for', '127.0.0.1').split(',')[0].strip()
        )

        status_code, body_data, extra_headers = process_earnings_request(symbol, client_ip)

        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        for k, v in extra_headers.items():
            self.send_header(k, v)
        self.end_headers()

        self.wfile.write(json.dumps(body_data).encode('utf-8'))
