"""
Point d'entrée principal pour Vercel Python Runtime.
Contient l'application WSGI 'app' avec rate limiting (5 req/jour/IP par route) et cache serveur 24h.

Routes :
- /api/earnings?symbol=AAPL               -> Alpha Vantage EARNINGS (clé ALPHAVANTAGE_API_KEY)
- /api/prices?type=prices&symbol=AAPL     -> Tiingo cours quotidiens (clé TIINGO_API_KEY)
"""

from http.server import BaseHTTPRequestHandler
import urllib.request
import urllib.parse
import json
import time
import os
import re

IP_REQUESTS = {}  # { (route, ip): [timestamp1, timestamp2, ...] }
DATA_CACHE = {}   # { (route, symbol): { "data": dict | list, "timestamp": float } }

DAILY_LIMIT = 5
WINDOW_SECONDS = 86400  # 24 heures glissantes
TIINGO_START_DATE = "2000-01-01"
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,10}$")


def fetch_alphavantage(symbol: str):
    """Appelle Alpha Vantage. Retourne (données, None) ou (None, (code, erreur))."""
    api_key = os.environ.get('ALPHAVANTAGE_API_KEY')
    if not api_key:
        return None, (500, {
            "error": "Configuration serveur manquante",
            "message": "La variable ALPHAVANTAGE_API_KEY n'est pas configurée dans les paramètres Vercel."
        })
    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={api_key}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Vercel-Proxy/1.0'})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        return None, (502, {"error": "Erreur lors de l'appel à Alpha Vantage", "details": str(e)})
    return data, None


def fetch_tiingo(symbol: str):
    """Appelle Tiingo (historique complet). Retourne (données, None) ou (None, (code, erreur))."""
    api_key = os.environ.get('TIINGO_API_KEY') or os.environ.get('TINGO_API_KEY')
    if not api_key:
        return None, (500, {
            "error": "Configuration serveur manquante",
            "message": "La variable TIINGO_API_KEY n'est pas configurée dans les paramètres Vercel."
        })
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate={TIINGO_START_DATE}"
    try:
        req = urllib.request.Request(url, headers={
            'Content-Type': 'application/json',
            'Authorization': f'Token {api_key}',
            'User-Agent': 'Vercel-Proxy/1.0'
        })
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        return None, (502, {"error": "Erreur lors de l'appel à Tiingo", "details": str(e)})
    if not isinstance(data, list):
        return None, (502, {"error": "Réponse inattendue de Tiingo", "details": str(data)[:300]})
    return data, None


ROUTES = {
    "earnings": {"fetch": fetch_alphavantage, "is_valid": lambda d: isinstance(d, dict) and "quarterlyEarnings" in d},
    "prices": {"fetch": fetch_tiingo, "is_valid": lambda d: isinstance(d, list) and len(d) > 0},
}


def resolve_route(path: str, params: dict) -> str:
    """Détermine la route demandée (par le paramètre ?type= ou, à défaut, par le chemin)."""
    route_type = params.get('type', [''])[0].lower()
    if route_type in ROUTES:
        return route_type
    if 'prices' in (path or ''):
        return 'prices'
    return 'earnings'


def process_request(route: str, symbol: str, client_ip: str):
    """Logique de traitement avec rate limit et cache."""
    if not SYMBOL_PATTERN.match(symbol):
        return 400, {"error": "Symbole invalide", "symbol": symbol}, {}

    now = time.time()
    rate_key = (route, client_ip)
    timestamps = IP_REQUESTS.get(rate_key, [])
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
    cache_key = (route, symbol)
    cached = DATA_CACHE.get(cache_key)
    if cached and (now - cached['timestamp'] < WINDOW_SECONDS):
        response_data = cached['data']
    else:
        response_data, error = ROUTES[route]["fetch"](symbol)
        if error:
            return error[0], error[1], {}
        if not ROUTES[route]["is_valid"](response_data):
            # Ex. message de quota Alpha Vantage renvoyé avec un statut 200 : on le signale comme
            # une erreur (non mise en cache, non décomptée du quota) plutôt que de le transmettre
            return 502, {"error": "Réponse inattendue de la source de données",
                         "details": str(response_data)[:300]}, {}
        DATA_CACHE[cache_key] = {"data": response_data, "timestamp": now}

    valid_timestamps.append(now)
    IP_REQUESTS[rate_key] = valid_timestamps
    remaining = max(0, DAILY_LIMIT - len(valid_timestamps))

    headers = {
        'Cache-Control': 'public, s-maxage=43200, stale-while-revalidate=21600',
        'X-RateLimit-Limit': str(DAILY_LIMIT),
        'X-RateLimit-Remaining': str(remaining)
    }
    return 200, response_data, headers


STATUS_TEXT = {
    200: '200 OK',
    400: '400 Bad Request',
    429: '429 Too Many Requests',
    500: '500 Internal Server Error',
    502: '502 Bad Gateway'
}


# 1. Point d'entrée WSGI principal nommé 'app'
def app(environ, start_response):
    method = environ.get('REQUEST_METHOD', 'GET')
    if method == 'OPTIONS':
        start_response('200 OK', [
            ('Access-Control-Allow-Origin', '*'),
            ('Access-Control-Allow-Methods', 'GET, OPTIONS'),
            ('Access-Control-Allow-Headers', 'Content-Type')
        ])
        return [b'']

    params = urllib.parse.parse_qs(environ.get('QUERY_STRING', ''))
    symbol = params.get('symbol', ['AAPL'])[0].upper()
    route = resolve_route(environ.get('PATH_INFO', ''), params)

    client_ip = (
        environ.get('HTTP_X_REAL_IP')
        or environ.get('HTTP_X_FORWARDED_FOR', '127.0.0.1').split(',')[0].strip()
    )

    status_code, body_data, extra_headers = process_request(route, symbol, client_ip)

    resp_headers = [
        ('Content-Type', 'application/json; charset=utf-8'),
        ('Access-Control-Allow-Origin', '*')
    ]
    for k, v in extra_headers.items():
        resp_headers.append((k, v))

    start_response(STATUS_TEXT.get(status_code, f'{status_code} Error'), resp_headers)
    return [json.dumps(body_data).encode('utf-8')]


# Alias standard WSGI pour compatibilité Django/Gunicorn
application = app


# 2. Point d'entrée BaseHTTPRequestHandler pour compatibilité Vercel Serverless
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
        route = resolve_route(parsed_url.path, params)

        client_ip = (
            self.headers.get('x-real-ip')
            or self.headers.get('x-forwarded-for', '127.0.0.1').split(',')[0].strip()
        )

        status_code, body_data, extra_headers = process_request(route, symbol, client_ip)

        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        for k, v in extra_headers.items():
            self.send_header(k, v)
        self.end_headers()

        self.wfile.write(json.dumps(body_data).encode('utf-8'))
