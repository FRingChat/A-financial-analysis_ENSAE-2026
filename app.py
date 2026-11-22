"""
Point d'entrée principal pour Vercel Python Runtime.
Exporte 'app' (WSGI) et 'handler' (BaseHTTPRequestHandler) depuis api/index.py.
"""
import sys
import os

sys.path.append(os.path.dirname(__file__))

from api.index import app, handler
