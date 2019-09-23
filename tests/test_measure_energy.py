import pytest
from microquake.processors.measure_energy import Processor

pytest.test_data_name = "test_output_focal_mechanism"


def test_measure_energy(catalog, waveform_stream):
    processor = Processor()
    processor.process(cat=catalog, stream=waveform_stream)
    output_catalog = processor.output_catalog(catalog)

    check_measure_energy_data(output_catalog)


def check_measure_energy_data(output_catalog):
    for event in output_catalog:
        origin = event.preferred_origin() if event.preferred_origin() else event.origins[0]

        for arr in origin.arrivals:
            assert abs(arr.vel_flux) > 0
            assert abs(arr.energy) > 0
