import os

import requests
from dynaconf import LazySettings, settings

from microquake.core import read_events
from microquake.core.stream import read


def get_test_data(file_name, data_format):
    test_data_location = settings.get("test_data_location")
    tmp_location = settings.get("local_temp_test_data_location")
    test_data_protocol_prefix = test_data_location.split("/")[0]
    # If location is remote, download the data and then open it
    remote_data_location = os.path.join(test_data_location, file_name)
    local_test_data_location = os.path.join(tmp_location, file_name)

    if test_data_protocol_prefix in ["http:", "https:"] and not os.path.isfile(local_test_data_location):
        resp = requests.get(remote_data_location)
        if resp.status_code != 200:
            raise Exception("Cannot load test data from: {}".format(remote_data_location))
        open(local_test_data_location, 'wb').write(resp.content)

    # if test_data_protocol_prefix == '':
    #     local_test_data_location = os.path.join(test_data_location, file_name)

    if not local_test_data_location:
        raise Exception("Cannot not find test data for {}".format(file_name))

    # >>> test_location = "/app/data/tests/test_output_picker.xml"
    if data_format == "MSEED":
        with open(local_test_data_location, "rb") as event_file:
            return read(event_file, format="MSEED")
    elif data_format == "QUAKEML":
        with open(local_test_data_location, "rb") as event_file:
            return read_events(event_file, format="QUAKEML")

def clean_test_data(file_name):
    test_data_location = settings.get("test_data_location")
    test_data_protocol_prefix = test_data_location.split("/")[0]
    # We only need to clean up data if we downloaded from remote
    if test_data_protocol_prefix not in ["http:", "https:"]:
        return
    tmp_location = settings.get("local_temp_test_data_location")
    local_test_data_location = os.path.join(tmp_location, file_name)
    if os.path.exists(local_test_data_location):
        os.remove(local_test_data_location)
