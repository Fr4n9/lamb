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


class TestLogTokenUsageCacheAware:
    def test_cost_with_cached_tokens(self, fresh_db):
        dm, _ = fresh_db
        conn = dm.get_connection()
        conn.execute("INSERT INTO organizations (id, name, slug, status, config, created_at, updated_at) VALUES (1, 'TestOrg', 'test-org', 'active', '{}', 1700000000, 1700000000)")
        conn.execute(
            "INSERT INTO assistants (id, name, owner, organization_id, api_callback, created_at, updated_at) VALUES (1, 'Bot', 'a@b.com', 1, '{}', 1700000000, 1700000000)"
        )
        conn.commit()
        conn.close()

        usage_data = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
            "prompt_tokens_details": {"cached_tokens": 800},
        }
        dm.log_token_usage(
            assistant_id=1, org_id=1, model_name="gpt-4o",
            provider="openai", usage_data=usage_data
        )

        conn = dm.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT cached_prompt_tokens_total, non_cached_prompt_tokens_total, prompt_tokens_total, cost_usd_total FROM assistant_usage_totals WHERE assistant_id = 1")
        row = cursor.fetchone()
        conn.close()

        assert row[0] == 800   # cached
        assert row[1] == 200   # non_cached
        assert row[2] == 1000  # total prompt
        # cost = (200 * 2.50/1e6) + (800 * 1.25/1e6) + (500 * 10.0/1e6)
        expected_cost = (200 * 2.50 / 1e6) + (800 * 1.25 / 1e6) + (500 * 10.0 / 1e6)
        assert abs(row[3] - expected_cost) < 1e-9

    def test_cost_without_cached_tokens_falls_back(self, fresh_db):
        dm, _ = fresh_db
        conn = dm.get_connection()
        conn.execute("INSERT INTO organizations (id, name, slug, status, config, created_at, updated_at) VALUES (1, 'TestOrg', 'test-org', 'active', '{}', 1700000000, 1700000000)")
        conn.execute(
            "INSERT INTO assistants (id, name, owner, organization_id, api_callback, created_at, updated_at) VALUES (2, 'Bot2', 'a@b.com', 1, '{}', 1700000000, 1700000000)"
        )
        conn.commit()
        conn.close()

        usage_data = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
        }
        dm.log_token_usage(
            assistant_id=2, org_id=1, model_name="gpt-4o",
            provider="openai", usage_data=usage_data
        )

        conn = dm.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT cached_prompt_tokens_total, non_cached_prompt_tokens_total, prompt_tokens_total, cost_usd_total FROM assistant_usage_totals WHERE assistant_id = 2")
        row = cursor.fetchone()
        conn.close()

        assert row[0] == 0     # no cached
        assert row[1] == 1000  # all non-cached
        assert row[2] == 1000
        expected_cost = (1000 * 2.50 / 1e6) + (500 * 10.0 / 1e6)
        assert abs(row[3] - expected_cost) < 1e-9

    def test_cost_no_pricing_row_returns_zero(self, fresh_db):
        dm, _ = fresh_db
        conn = dm.get_connection()
        conn.execute("INSERT INTO organizations (id, name, slug, status, config, created_at, updated_at) VALUES (1, 'TestOrg', 'test-org', 'active', '{}', 1700000000, 1700000000)")
        conn.execute(
            "INSERT INTO assistants (id, name, owner, organization_id, api_callback, created_at, updated_at) VALUES (3, 'Bot3', 'a@b.com', 1, '{}', 1700000000, 1700000000)"
        )
        conn.commit()
        conn.close()

        usage_data = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        dm.log_token_usage(
            assistant_id=3, org_id=1, model_name="unknown-model",
            provider="openai", usage_data=usage_data
        )

        conn = dm.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT cost_usd_total FROM assistant_usage_totals WHERE assistant_id = 3")
        row = cursor.fetchone()
        conn.close()
        assert row[0] == 0.0

    def test_usage_logs_stores_full_json(self, fresh_db):
        dm, _ = fresh_db
        conn = dm.get_connection()
        conn.execute("INSERT INTO organizations (id, name, slug, status, config, created_at, updated_at) VALUES (1, 'TestOrg', 'test-org', 'active', '{}', 1700000000, 1700000000)")
        conn.execute(
            "INSERT INTO assistants (id, name, owner, organization_id, api_callback, created_at, updated_at) VALUES (4, 'Bot4', 'a@b.com', 1, '{}', 1700000000, 1700000000)"
        )
        conn.commit()
        conn.close()

        usage_data = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 80},
            "extra_field": "preserved",
        }
        dm.log_token_usage(
            assistant_id=4, org_id=1, model_name="gpt-4o",
            provider="openai", usage_data=usage_data
        )

        import json
        conn = dm.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT usage_data FROM usage_logs WHERE assistant_id = 4")
        row = cursor.fetchone()
        conn.close()
        stored = json.loads(row[0])
        assert stored["prompt_tokens_details"]["cached_tokens"] == 80
        assert stored["extra_field"] == "preserved"

    def test_logging_failure_does_not_raise(self, fresh_db):
        dm, _ = fresh_db
        # Should not raise even with invalid assistant_id (FK might not be enforced in SQLite by default)
        dm.log_token_usage(
            assistant_id=99999, org_id=1, model_name="x",
            provider="x", usage_data={}
        )


# Shared fixture for admin API tests — patches security + verify_admin_access
@pytest.fixture
def admin_client(monkeypatch):
    """TestClient with admin auth bypassed for organization_router endpoints.

    Routes use dependencies=[Depends(security)] + manual verify_admin_access(request),
    NOT get_auth_context. We must patch both.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from creator_interface.organization_router import router
    from creator_interface import organization_router as org_router

    async def _noop_verify(request):
        return "test-token"

    monkeypatch.setattr(org_router, "verify_admin_access", _noop_verify)

    app = FastAPI()
    app.include_router(router, prefix="/admin")
    app.dependency_overrides[org_router.security] = lambda: {"credentials": "test-token"}

    return TestClient(app)


class TestCostOverviewAPI:
    def test_cost_overview_includes_summary(self, admin_client, monkeypatch):
        from creator_interface import organization_router as org_router

        mock_rows = [
            {
                "id": 1, "name": "Bot", "owner": "a@b.com",
                "organization_name": "TestOrg", "organization_id": 1,
                "api_callback": '{"llm": "gpt-4o", "connector": "openai"}',
                "prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500,
                "cost_usd": 0.01, "thresholds_config": None,
                "cached_prompt_tokens": 800, "non_cached_prompt_tokens": 200,
            }
        ]
        mock_db = MagicMock()
        mock_db.get_all_assistants_with_usage.return_value = mock_rows
        monkeypatch.setattr(org_router, "db_manager", mock_db)

        resp = admin_client.get("/admin/cost-overview", headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert "assistants" in data
        assert data["assistants"][0]["organization_id"] == 1
        assert data["assistants"][0]["cached_prompt_tokens"] == 800
        assert data["assistants"][0]["non_cached_prompt_tokens"] == 200
        assert "cache_hit_percentage" in data["assistants"][0]
        assert data["summary"]["total_cost_usd"] == 0.01
        assert data["summary"]["cached_prompt_tokens"] == 800


class TestUsageByModelAPI:
    def test_usage_by_model_returns_breakdown(self, admin_client, monkeypatch):
        from creator_interface import organization_router as org_router

        mock_db = MagicMock()
        mock_db.get_assistant_usage_by_model.return_value = [
            {
                "provider": "openai",
                "model_name": "gpt-4o",
                "prompt_tokens": 12000,
                "cached_prompt_tokens": 9000,
                "non_cached_prompt_tokens": 3000,
                "completion_tokens": 8000,
                "total_tokens": 20000,
                "cost_usd": 0.42,
                "request_count": 85,
                "input_per_1m": 2.50,
                "cached_input_per_1m": 1.25,
                "output_per_1m": 10.0,
            }
        ]
        mock_db.get_assistant_by_id.return_value = MagicMock()
        monkeypatch.setattr(org_router, "db_manager", mock_db)

        resp = admin_client.get("/admin/assistant/10/usage-by-model", headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["assistant_id"] == 10
        assert len(data["breakdown"]) == 1
        assert data["breakdown"][0]["model_name"] == "gpt-4o"
        assert data["breakdown"][0]["pricing"]["cached_input_per_1m"] == 1.25


class TestOrgSearchAPI:
    def test_org_search_returns_matches(self, admin_client, monkeypatch):
        from creator_interface import organization_router as org_router
        mock_db = MagicMock()
        mock_db.search_organizations.return_value = [
            {"id": 3, "name": "PEPESITO", "slug": "pepesito"}
        ]
        monkeypatch.setattr(org_router, "db_manager", mock_db)

        resp = admin_client.get("/admin/organizations/search?name=pepe", headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["organizations"]) == 1
        assert data["organizations"][0]["name"] == "PEPESITO"

    def test_org_summary_scoped_to_org(self, admin_client, monkeypatch):
        from creator_interface import organization_router as org_router
        mock_db = MagicMock()
        mock_db.get_org_scoped_summary.return_value = {
            "total_cost_usd": 0.5,
            "total_tokens": 1000,
            "prompt_tokens": 600,
            "completion_tokens": 400,
            "cached_prompt_tokens": 300,
            "assistant_count": 2,
            "quota_exceeded_count": 0,
        }
        monkeypatch.setattr(org_router, "db_manager", mock_db)

        resp = admin_client.get("/admin/cost-overview/summary?organization_id=3", headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["total_cost_usd"] == 0.5
        assert data["summary"]["assistant_count"] == 2
