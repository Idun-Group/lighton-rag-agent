import os

import pytest
from dotenv import load_dotenv

from sdk import LightOn


@pytest.fixture
def mock_client():
    """Client pointed at a fake base URL for unit tests."""
    return LightOn(api_key="fake-key", base_url="https://fake.lighton.ai")


@pytest.fixture(scope="session")
def live_client():
    """Client using the real API key from .env for integration tests."""
    load_dotenv()
    api_key = os.environ.get("LIGHTON_API_KEY")
    if not api_key:
        pytest.skip("LIGHTON_API_KEY not set")
    client = LightOn(api_key=api_key)
    yield client
    client.close()
