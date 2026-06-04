"""Tests for cost management — cache-aware token costs & model breakdown."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_BACKEND_ROOT = Path(__file__).parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Create a temporary SQLite DB and run all migrations.

    LambDatabaseManager.__init__ reads config.LAMB_DB_PATH and takes no args,
    so we monkeypatch the config module before instantiation.
    database_manager.py imports config as a top-level module (not lamb.config).
    
    We also patch initialize_system_organization() to avoid OWI dependency in CI.
    """
    import config
    monkeypatch.setattr(config, "LAMB_DB_PATH", str(tmp_path))
    monkeypatch.setattr(config, "LAMB_DB_PREFIX", "")
    from lamb.database_manager import LambDatabaseManager
    monkeypatch.setattr(LambDatabaseManager, "initialize_system_organization", lambda self: None)
    dm = LambDatabaseManager()
    yield dm, str(tmp_path / "lamb_v4.db")


class TestMigration18:
    def test_model_pricing_has_cached_input_column(self, fresh_db):
        dm, _ = fresh_db
        conn = dm.get_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(model_pricing)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        assert "cached_input_per_1m" in columns

    def test_assistant_usage_totals_has_cache_columns(self, fresh_db):
        dm, _ = fresh_db
        conn = dm.get_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(assistant_usage_totals)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        assert "cached_prompt_tokens_total" in columns
        assert "non_cached_prompt_tokens_total" in columns

    def test_model_pricing_seed_includes_cached_rates(self, fresh_db):
        dm, _ = fresh_db
        conn = dm.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT model_name, cached_input_per_1m FROM model_pricing WHERE provider = 'openai' AND model_name = 'gpt-4o'"
        )
        row = cursor.fetchone()
        conn.close()
        assert row is not None
        assert row[1] is not None
        assert row[1] > 0
