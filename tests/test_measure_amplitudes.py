import pytest
from microquake.processors.measure_amplitudes import Processor
from obspy import UTCDateTime

pytest.test_data_name = "test_output_nlloc"


def test_measure_amplitudes(catalog, waveform_stream):
    processor = Processor()
    res = processor.process(cat=catalog, stream=waveform_stream)

    check_amplitudes_data(res['cat'])


def check_amplitudes_data(output_catalog):
    for event in output_catalog:
        origin = event.preferred_origin() if event.preferred_origin() else event.origins[0]

        for arr in origin.arrivals:
            if arr.polarity is not None:
                assert UTCDateTime(arr.t1) is not None
                assert UTCDateTime(arr.t2) is not None
                assert abs(arr.peak_vel) > 0
                assert UTCDateTime(arr.tpeak_vel) is not None
                assert abs(arr.pulse_snr) > 0

                if arr.peak_dis is not None:
                    assert abs(arr.peak_dis) > 0
                    assert abs(arr.max_dis) > 0
                    assert UTCDateTime(arr.tpeak_dis) is not None
                    assert UTCDateTime(arr.tmax_dis) is not None

                if arr.dis_pulse_area is not None:
                    assert arr.dis_pulse_area > 0

                if arr.dis_pulse_width is not None:
                    assert arr.dis_pulse_width > 0
