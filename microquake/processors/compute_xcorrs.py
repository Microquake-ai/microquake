import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from xseis2 import xchange, xio
from xseis2.h5stream import H5Stream
from xseis2.xsql import XCorr

import redis
from loguru import logger

from microquake.core.settings import settings
from microquake.processors.processing_unit import ProcessingUnit


class Processor(ProcessingUnit):
    @property
    def module_name(self):
        return "compute_xcorrs"

    def initializer(self):

        # Mock store for raw data (replace with database handle later)

        hfname = os.path.join(settings.common_dir, 'ot_cont', '10hr_sim.h5')
        self.rawdata_db = H5Stream(hfname)

        # sql database to store cross-correlation metadata
        db_string = "postgres://postgres:postgres@localhost"
        db = create_engine(db_string)
        print(db.table_names())
        Session = sessionmaker(bind=db)
        self.session = Session()

        # redis store for cross-correlation data
        self.rhandle = redis.Redis(host='localhost', port=6379, db=0)

    def process(self, **kwargs):
        logger.info("pipeline: compute_xcorrs")

        t0 = kwargs['t0']
        t1 = kwargs['t1']

        # channels to correlate can either be (1) taken from db (2) or params
        chans = self.rawdata_db.channels
        sr_raw = self.rawdata_db.samplerate
        # chans = self.params.channels
        # chans = kwargs['chans']

        # params
        cclen = self.params.cclen_sec
        stepsize = self.params.stepsize
        onebit = self.params.onebit
        whiten_freqs = self.params.whiten_corner_freqs
        sr = self.params.cc_samplerate
        keeplag = self.params.keeplag

        # python generator which yields slices of data
        datgen = self.rawdata_db.slice_gen(t0, t1, chans, cclen, stepsize=stepsize)

        dc, ckeys_ix = xchange.xcorr_stack_slices(
            datgen, chans, cclen, sr_raw, sr, keeplag, whiten_freqs, onebit=onebit)

        ckeys = [f"{ck[0]}_{ck[1]}" for ck in chans[ckeys_ix]]

        pipe = self.rhandle.pipeline()

        rows = []

        for i, sig in enumerate(dc):
            # print(i)
            ckey = ckeys[i]
            dkey = f"{str(t0)} {ckey}"
            pipe.set(dkey, xio.array_to_bytes(sig))
            rows.append(XCorr(time=t0, ckey=ckey, data=dkey))

        pipe.execute()  # add data to redis
        self.session.add_all(rows)  # add rows to sql
        self.session.commit()

        return
