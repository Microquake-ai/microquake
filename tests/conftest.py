from os import path

import pytest

from fakeredis import FakeStrictRedis


@pytest.fixture(scope="session", autouse=True)
def redis():
    return FakeStrictRedis()


@pytest.fixture(scope="session", autouse=True)
def picks():
    pick_path = path.join(path.dirname(path.realpath(__file__)), 'fixtures/picks.json')
    with open(pick_path) as file:
        data = file.read()

        return data
