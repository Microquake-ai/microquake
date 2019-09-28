import pytest
from microquake.processors.focal_mechanism import Processor

pytest.test_data_name = "test_output_smom"


def test_focal_mechanism(catalog, waveform_stream):
    processor = Processor()
    res = processor.process(cat=catalog, stream=waveform_stream)

    check_focal_mechanism_data(res['cat'])


def check_focal_mechanism_data(output_catalog):
    for event in output_catalog:
        assert len(event.focal_mechanisms) > 0
        assert event.preferred_focal_mechanism_id

        for focal_mechanism in event.focal_mechanisms:
            assert focal_mechanism.station_polarity_count > 0
