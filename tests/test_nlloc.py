import pytest
from microquake.processors.nlloc import Processor
from obspy.core.event import OriginUncertainty

pytest.test_data_name = "test_output_picker"


def test_hypocenter_location(catalog, waveform_stream):
    processor = Processor()
    processor.process(cat=catalog, stream=waveform_stream)
    output_catalog = processor.output_catalog(catalog)

    check_hypocenter_location(catalog, output_catalog)


def check_hypocenter_location(input_catalog, output_catalog):
    assert input_catalog[0].preferred_origin().origin_uncertainty is None
    origin_uncertainty = output_catalog[0].preferred_origin().origin_uncertainty
    assert isinstance(origin_uncertainty, OriginUncertainty)
    assert origin_uncertainty.confidence_ellipsoid.semi_major_axis_length > 0

    assert origin_uncertainty.confidence_ellipsoid.semi_minor_axis_length > 0

    assert origin_uncertainty.confidence_ellipsoid.semi_intermediate_axis_length > 0

    origin = output_catalog[0].preferred_origin()

    for arr in origin.arrivals:
        assert arr.hypo_dist_in_m == arr.distance
