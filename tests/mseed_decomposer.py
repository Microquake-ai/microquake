from importlib import reload
import numpy as np
import os
import glob
from microquake.core import read
from io import BytesIO
from spp.utils.application import Application
# from struct import unpack
# from datetime import datetime

# from microquake.core.util import tools
from microquake.io import waveform
import matplotlib.pyplot as plt
plt.ion()

app = Application()

# write a stream to mseed buffer
st = read(os.path.join(app.common_dir, "synthetic", "simdat10s.mseed"))
st.traces = st.traces[:2]
buf = BytesIO()
st.write(buf, format='MSEED')

dchunks = waveform.decompose_mseed(buf.getvalue())

# reconstruct mseed from dchunks
dat = b''
for k, v in dchunks.items():
    dat += v

# compare original and reconstructed data
st2 = read(BytesIO(dat))

tr = st[1]
tr2 = st2[1]
# plt.plot(tr.data)
# plt.plot(tr2.data)
print("all_close: ", np.allclose(tr.data, tr2.data))
