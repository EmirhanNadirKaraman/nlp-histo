"""
Database package for NLP Histopathology project.
Provides models and connection utilities for PostgreSQL.
"""

from .models import Document, TextElement, Figure, Base
from .db_connection import (
    DatabaseConnection,
    get_db_connection,
    close_db_connection,
    get_database_url,
    DB_CONFIG
)

__all__ = [
    'Document',
    'TextElement',
    'Figure',
    'Base',
    'DatabaseConnection',
    'get_db_connection',
    'close_db_connection',
    'get_database_url',
    'DB_CONFIG'
]
