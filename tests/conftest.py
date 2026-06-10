import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_weather_data import create_fake_weather_root


@pytest.fixture()
def fake_weather_root(tmp_path):
    return create_fake_weather_root(tmp_path)
