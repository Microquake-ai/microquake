import pytest
from microquake.processors.measure_smom import Processor

pytest.test_data_name = "test_output_amplitude"


def test_measure_smom(catalog, waveform_stream):
    processor = Processor()
    processor.process(cat=catalog, stream=waveform_stream)

    check_smom_data(catalog)


def check_smom_data(output_catalog):
    for event in output_catalog:
        for arr in event.preferred_origin().arrivals:
            if arr.smom:
                assert abs(arr.smom) > 0
                assert abs(arr.fit) > 0
                assert abs(arr.tstar) > 0
