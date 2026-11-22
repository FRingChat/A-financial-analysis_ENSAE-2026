"""
Module de compatibilité api/index.py
Redirige vers app.py
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import app, application, handler
