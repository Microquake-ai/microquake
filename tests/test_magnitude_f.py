import pytest
from microquake.processors.magnitude import Processor

pytest.test_data_name = "test_output_magnitude"


def test_magnitude_f(catalog):
    processor = Processor(module_type="frequency")
    res = processor.process(cat=catalog.copy())

    check_magnitude_f_data(catalog, res['cat'])


def check_magnitude_f_data(input_catalog, output_catalog):
    input_magnitude_f_count = len(input_catalog[0].magnitudes)

    event = output_catalog[0]
    assert event.magnitudes
    assert len(event.magnitudes) > input_magnitude_f_count
    assert event.station_magnitudes
