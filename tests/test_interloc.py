import pytest
from microquake.core.event import Origin
from microquake.processors.interloc import Processor

pytest.test_data_name = "test_end_to_end"


def test_interloc(catalog, waveform_stream):
    processor = Processor()
    processor.process(stream=waveform_stream)
    output_catalog = processor.output_catalog(catalog)

    check_interloc_data(catalog, output_catalog)


def check_interloc_data(input_catalog, output_catalog):
    original_origin_count = len(input_catalog[0].origins)

    assert len(output_catalog[0].origins) == (original_origin_count + 1)
    assert isinstance(output_catalog[0].origins[0], Origin)
    assert output_catalog[0].preferred_origin_id is not None

    assert output_catalog[0].preferred_origin().extra.interloc_normed_vmax.value is not None
    assert output_catalog[0].preferred_origin().extra.interloc_normed_vmax.namespace == 'MICROQUAKE'

# Some interloc data is removed later in the pipeline


def check_interloc_data_end_to_end(output_catalog):
    assert isinstance(output_catalog[0].origins[0], Origin)
    assert output_catalog[0].preferred_origin_id is not None
