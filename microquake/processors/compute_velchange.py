import numpy as np

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from xseis2 import xchange, xio
from xseis2.xsql import XCorr, ChanPair

import redis
from loguru import logger

from microquake.processors.processing_unit import ProcessingUnit


class Processor(ProcessingUnit):
    @property
    def module_name(self):
        return "compute_velchange"

    def initializer(self):

        # sql database of cross-correlation metadata
        db_string = "postgres://postgres:postgres@localhost"
        db = create_engine(db_string)
        print(db.table_names())
        Session = sessionmaker(bind=db)
        self.session = Session()

        # redis store for cross-correlation data
        self.rhandle = redis.Redis(host='localhost', port=6379, db=0)

    def process(self, **kwargs):
        logger.info("pipeline: compute_xcorrs")

        session = self.session
        rhandle = self.rhandle
        # params
        sr = self.params.cc_samplerate
        wlen_sec = self.params.wlen_sec
        coda_start_vel = self.params.coda_start_velocity
        coda_end_sec = self.params.coda_end_sec
        whiten_freqs = self.params.whiten_corner_freqs
        nrecent = 10

        ckeys = np.unique(session.query(XCorr.ckey).all())

        for ckey in ckeys[:]:

            dist = session.query(ChanPair.dist).filter(ChanPair.ckey == ckey).first()[0]
            coda_start_sec = dist / coda_start_vel
            print(f"{ckey}: {dist:.2f}m")

            ccfs = session.query(XCorr).filter_by(ckey=ckey).order_by(XCorr.time.desc()).limit(nrecent).all()[::-1]

            for icc in range(1, len(ccfs)):
                # print(i)
                cc_ref = ccfs[icc - 1]
                cc_cur = ccfs[icc]
                sig_ref = xio.bytes_to_array(rhandle.get(cc_ref.data))
                sig_cur = xio.bytes_to_array(rhandle.get(cc_cur.data))
                dvv, error = xchange.dvv(sig_ref, sig_cur, sr, wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec)
                cc_cur.dvv = dvv
                cc_cur.error = error

        session.commit()
