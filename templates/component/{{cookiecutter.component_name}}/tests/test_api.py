"""Tests for {{cookiecutter.component_name}} API endpoints."""

import pytest
from fastapi.testclient import TestClient
from {{cookiecutter.component_name.replace('-', '_')}}.api.main import app


@pytest.fixture
def client() -> TestClient:
    """Create test client.

    Returns:
        FastAPI test client.
    """
    return TestClient(app)


class TestAPI:
    """Test suite for API endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_process_endpoint(self, client: TestClient) -> None:
        """Test process endpoint."""
        response = client.post(
            "/api/v1/process",
            json={"data": {"test": "input"}}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
