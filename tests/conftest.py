from os import environ, getcwd, path

import pytest
from fakeredis import FakeStrictRedis

from microquake.core.settings import settings

from .helpers.data_utils import get_test_data

pytest.test_data_name = None


@pytest.fixture(scope="session", autouse=True)
def api_url():
    return settings.get('api_base_url')


@pytest.fixture(scope="session", autouse=True)
def picks():
    pick_path = path.join(path.dirname(path.realpath(__file__)), 'fixtures/picks.json')
    with open(pick_path) as file:
        data = file.read()

        return data


@pytest.fixture(scope="function")
def catalog():
    file_name = pytest.test_data_name + ".xml"

    return get_test_data(file_name, "QUAKEML")


@pytest.fixture(scope="function")
def waveform_stream():
    file_name = pytest.test_data_name + ".mseed"

    return get_test_data(file_name, "MSEED")


@pytest.fixture(scope="session", autouse=True)
def redis():
    return FakeStrictRedis()
