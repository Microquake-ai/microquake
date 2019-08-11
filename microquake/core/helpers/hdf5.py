from os import path

from microquake.core.data import ttable

from microquake.core.settings import settings


def get_ttable_h5():
    fname = path.join(settings.common_dir,
                      settings.grids.travel_time_h5.fname)

    return ttable.H5TTable(fname)


def write_ttable_h5(fname=None):
    nll_tts_dir = path.join(settings.nll_base, 'time')

    if fname is None:
        fname = settings.grids.travel_time_h5.fname

    ttp = ttable.array_from_nll_grids(nll_tts_dir, 'P', prefix='OT')
    tts = ttable.array_from_nll_grids(nll_tts_dir, 'S', prefix='OT')
    fpath = path.join(settings.common_dir, fname)
    ttable.write_h5(fpath, ttp, tdict2=tts)
