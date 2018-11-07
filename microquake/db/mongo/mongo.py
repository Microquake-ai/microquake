# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: mongo.py
#  Purpose: module to interract with mongo db
#   Author: microquake development team
#    Email: dev@microquake.org
#
# Copyright (C) 2016 microquake development team
# --------------------------------------------------------------------
"""
module to interract with mongodb

:copyright:
    microquake development team (dev@microquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""


import numpy as np
from datetime import datetime
from microquake.core.util.attribdict import AttribDict
import json
from bson import ObjectId
from obspy.core.utcdatetime import UTCDateTime
from pymongo.bulk import BulkWriteError
#from spp.utils import log_handler

#log = log_handler.get_logger("mongoDB", 'spp_api.log')


class MongoJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


class MongoDBHandler:

    db = None

    def __init__(self, uri='mongodb://localhost:27017/', db_name='test'):
        if self.db is None:
            #log.info("MongoDBHandler ------------> Initiating new DB connection")
            print("MongoDBHandler ------------> Initiating new DB connection")
            self.db = self.connect(uri, db_name)

    def connect(self, uri='mongodb://localhost:27017/', db_name='test'):
        """
        :param uri: uniform resource identifier
        :type uri: str
        :param db_name: database name
        :type db_name: str
        :return: pymongo db object
        """
        from pymongo import MongoClient
        client = MongoClient(uri)
        db = client[db_name]
        return db

    def disconnect(self):
        self.db.client.close()

    def insert_or_update(self, collection_name, document, filter_dictionary):
        """
        Write a microquake.core.event catalog into the project. Microquake
        only one event per catalog which are then stored in individual QuakeML
        files.
        :param event: event
        :type event: microquake.core.event.Event
        :type base_file_name: str
        :param waveform: microquake.core.Stream object (default = None)
        :rparam: id of inserted object
        """
        cursor = self.db[collection_name].find(filter_dictionary)

        if cursor.count():
            inserted_id = self.db[collection_name].replace_one(filter_dictionary, document)
        else:
            inserted_id = self.db[collection_name].insert_one(document)
        return inserted_id

    def select(self, collection_name, filter_dictionary):
        return self.db[collection_name].find_one(filter_dictionary)

    def insert_many(self, collection_name, documents):
        # Handling the errors raised from inserting duplicates
        try:
            self.db[collection_name].insert_many(documents, ordered=False)
        except BulkWriteError as bwe:
            for we in bwe.details['writeErrors']:
                print('Insertion Error:', we['errmsg'])



class EventDB:

    def __init__(self,mongodb_handler):
        self.DB_COLLECTION = "Events"
        self.DB_CONNECTION = mongodb_handler

    @staticmethod
    def flatten_event(event):
        """
        Extract essential information from an event
        :param event: an event object
        :type event: ~microquake.core.event.Event
        :return: return a dictionary with a selection of event properties
        """

        ev = {}
        if event.preferred_origin():
            origin = event.preferred_origin()
        elif event.origins:
            origin = event.origins[-1]
        else:
            origin = None

        if event.preferred_magnitude():
            magnitude = event.preferred_magnitude()
        elif event.magnitudes:
            magnitude = event.magnitudes[-1]
        else:
            magnitude = None

        if origin:
            ev['x'] = origin.x
            ev['y'] = origin.y
            ev['z'] = origin.z
            ev['time'] = int(np.float64(UTCDateTime(origin.time.datetime).timestamp) * 1e9)
            if origin.origin_uncertainty:
                if origin.origin_uncertainty.confidence_ellipsoid:
                    confidence_ellipsoid = \
                        origin.origin_uncertainty.confidence_ellipsoid
                    ev['uncertainty'] = confidence_ellipsoid.semi_major_axis_length
            ev['evaluation_mode'] = origin.evaluation_mode
            ev['status'] = getattr(origin, 'evaluation_status', 'not_picked')
            ev['event_type'] = getattr(event, 'event_type', 'not_reported')
            ev['event_resource_id'] = event.resource_id.id

            sqr = 0
            ct = 0
            for arrival in origin.arrivals:
                if not arrival:
                    continue
                ct += 1
# MTH: not every arrival has a time_residual!
                #sqr += arrival.time_residual ** 2
            #rms = np.sqrt(np.mean(sqr))
            rms = -9.

            ev['time_residual'] = rms
            ev['npick'] = len(origin.arrivals)

        if magnitude:
            ev['magnitude'] = magnitude.mag
            ev['magnitude_type'] = magnitude.magnitude_type
        else:
            ev['magnitude'] = None

        #ev['encoded_quakeml'] = EventDB.encode_event(event)

        return ev

    @staticmethod
    def encode_event(event):
        """
        Encode an event for storage in the database
        :param event: a microquake.core.event object
        :return: encoded event object in compressed (bz2) QuakeML format
        """
        from microquake.core.event import Catalog
        from io import BytesIO
        from microquake.core.util import serializer

        buf = BytesIO()
        cat = Catalog(events=[event])
        cat.write(buf, format='QUAKEML')

        return serializer.encode_base64(buf)

    @staticmethod
    def decode_event(encoded_event):
        """
        decode an event stored in the database
        :param encoded_event: compressed serialized object stored in the DB
        :return: microquake.core.event.event
        """

        from microquake.core import read_events
        from microquake.core.util import serializer

        cat = read_events(serializer.decode_base64(encoded_event))
        return cat[0]

    def insert_event(self, event, catalog_index=0):
        """
        Write a microquake.core.event catalog into the project. Microquake
        only one event per catalog which are then stored in individual QuakeML
        files.
        :param event: event
        :type event: microquake.core.event.Event
        :type base_file_name: str
        :param waveform: microquake.core.Stream object (default = None)
        :rparam: id of inserted object
        """
        ev_dict = self.flatten_event(event)
        ev_dict['catalog_index'] = catalog_index
        ev_dict['modification_time'] = datetime.now()
        # the index of the event in
        #  the catalog. Should in principle always be 0. Included for future
        # proofing purposes

        filter_dict = {'event_resource_id': event.resource_id.id}

        inserted_id = self.DB_CONNECTION.insert_or_update(self.DB_COLLECTION, ev_dict, filter_dict)

        return inserted_id


    # return json event object and contains its full encoded object
    def read_event(self, event_resource_id):
        filter_dict = {"event_resource_id": event_resource_id}
        return self.DB_CONNECTION.select(self.DB_COLLECTION, filter_dict)


    # return a microquake Event object after decoding it from DB
    def read_full_event(self, event_resource_id):
        result = self.read_event(event_resource_id)
        if result and result["encoded_quakeml"]:
            decoded_event = self.decode_event(result["encoded_quakeml"])
            return decoded_event
        else:
            return None


class StreamDB:

    def __init__(self, mongodb_handler):
        self.DB_COLLECTION = "Waveforms"
        self.DB_CONNECTION = mongodb_handler

    @staticmethod
    def encode_stream(stream):
        """
        Encode a stream object for storage in the database
        :param stream: a microquake.core.Trace object to be encoded
        :return: encoded stream in bson format
        """

        from io import BytesIO
        from microquake.core.stream import Stream
        from microquake.core import UTCDateTime
        from microquake.core.util import serializer

        traces = []
        for tr in stream:
            trout = dict()
            trout['stats'] = dict()
            for key in tr.stats.keys():
                if isinstance(tr.stats[key], UTCDateTime):
                    trout['stats'][key] = tr.stats[key].datetime
                else:
                    trout['stats'][key] = tr.stats[key]

            stout = Stream(traces=[tr])
            buf = BytesIO()
            stout.write(buf, format='MSEED')
            trout['encoded_mseed'] = serializer.encode_base64(buf)
            traces.append(trout)

        return traces

    @staticmethod
    def decode_stream(encoded_stream):
        """
        Decode a stream object encoded in bson
        :param encoded_stream: encoded stream produced by the encode_stream function
        :return: a microquake.core.Stream object
        """

        from microquake.core.stream import Stream
        from microquake.core import read
        from io import BytesIO
        from microquake.core.util import serializer

        traces = []
        for encoded_tr in encoded_stream:
            bstream = serializer.decode_base64(encoded_tr['encoded_mseed'])
            tr = read(BytesIO(bstream))[0]
            traces.append(tr)

        return Stream(traces=traces)

    def insert_waveform(self, stream, event_resource_id):
        """
        :param db:
        :param stream:
        :param event:
        :return:
        """
        waveform_dict = dict()
        waveform_dict['event_resource_id'] = event_resource_id
        waveform_dict['waveform'] = self.encode_stream(stream)
        waveform_dict['modification_time'] = datetime.now()

        filter_dict = {'event_resource_id': event_resource_id}

        inserted_id = self.DB_CONNECTION.insert_or_update(self.DB_COLLECTION, waveform_dict, filter_dict )

        return inserted_id


    # return json waveform object and contains its full encoded object
    def read_waveform(self, event_resource_id):
        filter_dict = {'event_resource_id': event_resource_id}
        return self.DB_CONNECTION.select(self.DB_COLLECTION, filter_dict)


    # return a microquake waveform object after decoding it from DB
    def read_full_waveform(self, event_resource_id):
        result = self.read_waveform(event_resource_id)
        if result and result["waveform"]:
            decoded_event = self.decode_stream(result["waveform"])
            return decoded_event
        else:
            return None


###################################
# To be checked if will be used

def dict2Attribdict(obj):
    if isinstance(obj, dict):
        out = AttribDict()
        for key in obj.keys():
            out[key] = dict2Attribdict(obj[key])
        try:
            return out
        except:
            pass
    elif isinstance(obj, list):
        out = []
        for item in obj:
            out.append(dict2Attribdict(item))
        return out
    else:
        return obj


def AttribDict2dict(obj):
    if isinstance(obj, AttribDict):
        out = dict()
        for key in obj.keys():
            out[key] = AttribDict2dict(obj[key])
            return out
    elif isinstance(obj, list):
        out = []
        for item in obj:
            out.append(AttribDict2dict(item))
            return out
    else:
        return obj



# UNUSED FUNCTIONS #####

# def insert(col, obj, default=None, **kwargs):
#     """
#     Convert and insert an object into a mongo db collection
#     :param col: mongo db collection
#     :type catalog: pymongo.collection.Collection
#     :param obj: object to insert
#     :type obj: list of object or object
#     :param default: "default" function for JSONEncoder for ObsPy objects
#     :type default: microquake.core.json.default.Default
#     """
#     if not isinstance(obj, list):
#         obj = [obj]
#
#     tmp_list = []
#     # noinspection PyTypeChecker
#     for o in obj:
#         s = dumps(o, default=default, **kwargs)
#         tmp = loads(s)
#         tmp_list.append(tmp)
#
#     inserted_ids = col.insert_many(tmp_list).inserted_ids
#
#     # return [(id, obj) for id, obj in zip(tmp_list, inserted_ids)]
#     # return tmp_list, inserted_ids
#     return inserted_ids
#     # if not isinstance(obj, list):
#     #     obj = [obj]
#     #
#     # tmp_list = []





# def dump_event(event, db, mongo_col, data_path, base_file_name,
#                **kwargs):
#     """
#     Write a microquake.core.event catalog into a mongo db
#     :param event: event object
#     :type event: microquake.core.event.Event
#     :param db: mongodb database
#     :type db: pymongo.database.Database
#     :param mongo_col: collection name for events in db
#     :type mongo_col: str
#     :param data_path: path to raw data (seismogram and QuakeML catalog
#     :type data_path: str
#     :param base_file_name: base file name for the seismogram and the QuakeML
#     file. Basically the file name without the extension.
#     :type base_file_name: str
#     :rtype: bool
#     """
#
#     event_col = db[mongo_col]
#     default = Default()
#
#     # Skip if event already in DB
#     cursor = event_col.find_one({'base_file_name': base_file_name,
#         'data_path': data_path})
#     if cursor:
#         return None
#
#     event.data_path = data_path
#     event.base_file_name = base_file_name
#     event = flatten_event(event)
#
#     insert(event_col, event, default=default, **kwargs)




        # if tr.stats.segy:
        #     del tr.stats.segy
    #     trout['data'] = list(tr.data)
    #     stats_dict = dict(tr.stats)
    #     stats = dict()
    #     for key in stats_dict.keys():
    #         if key == 'starttime' or key == 'endtime':
    #             stats[key] = stats_dict[key].strftime(
    #         "%Y-%m-%d %H:%M%S.%f")
    #         elif key == 'mseed':
    #             mseed = dict()
    #             for m_key in stats_dict[key].keys():
    #                 mseed[m_key] = stats_dict[key][m_key]
    #             stats[key] = mseed
    #         else:
    #             stats[key] = stats_dict[key]
    #     trout['stats'] = stats
    #     traces.append(trout)
    #
    # stout['traces'] = traces

    # return b64encode(compress(dumps(stout).encode()))




    # tr_list = loads(decompress(b64decode(encoded_stream)))['traces']
    #
    # traces = []
    # for tr_dict in tr_list:
    #     tr = Trace()
    #     stats = Stats()
    #
    #     tr.data = array(tr_dict['data'])
    #     stats_dict = tr_dict['stats']
    #
    #     for key2 in stats_dict.keys():
    #         if (key2 == 'starttime'):
    #             stats[key2] = UTCDateTime().strptime(stats_dict[key2],
    #                                              "%Y-%m-%d %H:%M%S.%f")
    #
    #         elif (key2 == 'endtime'):
    #             continue
    #         else:
    #             stats[key2] = stats_dict[key2]
    #
    #     tr.stats = stats
    #
    #     traces.append(tr)
    #
    # return Stream(traces=traces)



