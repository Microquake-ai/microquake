import copy

def copy_picks_to_dict(picks):
    pick_dict = {}
    for pick in picks:
        station = pick.waveform_id.station_code
        phase   = pick.phase_hint
        if station not in pick_dict:
            pick_dict[station]={}
        pick_dict[station][phase]=copy.deepcopy(pick)
    return pick_dict
