import pytest
from microquake.processors.picker import Processor

pytest.test_data_name = "test_output_interloc"


def test_picker(catalog, waveform_stream):
    processor = Processor()
    processor.process(cat=catalog, stream=waveform_stream)
    output_catalog = processor.output_catalog(catalog)

    check_picker_data(catalog, output_catalog)


def check_picker_data(input_catalog, output_catalog):
    original_pick_count = len(input_catalog[0].picks)
    assert len(output_catalog[0].picks) > original_pick_count
