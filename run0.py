from __future__ import division
import numpy

from MAUD.core import Framework, Assembly, IndVar
from MAUD.solvers import *
from sumad import *
from MAUD.driver_pyoptsparse import *
from init_func import *

import sys

def redirectIO(f):
    """                                                                                                                                                                                                             
    Redirect stdout/stderr to the given file handle.                                                                                                                                                                
    Based on: http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/.                                                                                                                         
    Written by Bret Naylor                                                                                                                                                                                          
    """
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    # Flush and close sys.stdout/err - also closes the file descriptors (fd)                                                                                                                                        
    sys.stdout.close()
    sys.stderr.close()

    # Make original_stdout_fd point to the same file as to_fd                                                                                                                                                       
    os.dup2(f.fileno(), original_stdout_fd)
    os.dup2(f.fileno(), original_stderr_fd)

    # Create a new sys.stdout that points to the redirected fd                                                                                                                                                      
    sys.stdout = os.fdopen(original_stdout_fd, 'wb', 0) # 0 makes them unbuffered                                                                                                                                   
    sys.stderr = os.fdopen(original_stderr_fd, 'wb', 0)

filename = 'output%03i.out'%MPI.COMM_WORLD.rank
if MPI.COMM_WORLD.rank == 0:
    filename = 'output.out'
redirectIO(open(filename, 'w'))





A_list0 = []
M_list0 = []

A_list0.extend([3] * 3)
M_list0.extend([0.45, 0.6, 0.75])

A_list0.extend([-3] * 4)
M_list0.extend([0.45, 0.6, 0.75, 0.80])

A_list0.extend([0, -1, 1])
M_list0.extend([0.45, 0.80, 0.80])

A_list0.extend([-1] * 3)
M_list0.extend([0.68, 0.74, 0.78])

A_list0.extend([1] * 3)
M_list0.extend([0.68, 0.74, 0.78])

npt = len(A_list0)





execfile('options/euler_options.py')
execfile('options/sumb_options.py')
from mpi4py import MPI
func_dict = {}
aeroOptions['gridFile'] = 'fc00_224_vol.cgns'
meshOptions['gridFile'] = 'fc00_224_vol.cgns'

data_cl = numpy.zeros(npt)
data_cd = numpy.zeros(npt)

for ipt in xrange(npt):
    A = A_list0[ipt]
    M = M_list0[ipt]

    ap = AeroProblem(name='name', mach=M, altitude=10000,
                     areaRef=45.5, alpha=A, chordRef=3.25,
                     evalFuncs=['cl', 'cd'])
    CFDSolver = AEROSOLVER(options=aeroOptions, comm=MPI.COMM_WORLD)
    mesh = MBMesh(options=meshOptions, comm=MPI.COMM_WORLD)
    CFDSolver.setMesh(mesh)
    CFDSolver(ap)
    CFDSolver.evalFunctions(ap, func_dict)

    data_cl[ipt] = func_dict[ap['cl']]
    data_cd[ipt] = func_dict[ap['cd']]

    if MPI.COMM_WORLD.rank == 0:
        numpy.savetxt('D4pt_CL.dat', data_cl)
        numpy.savetxt('D4pt_CD.dat', data_cd)
