from pyoptsparse import SqliteDict
import sys
import numpy

num = 128
in_name = 'mrun/hist_0_%i.hst'
out_name = 'msn_profiles/msn_%i.dat'

for ind in xrange(num):
    db = SqliteDict(in_name%ind)
    data = db[db['last']]['xuser']['h_cp']
    numpy.savetxt(out_name%ind, data)
