""" OpenMDAO model that builds wraps and executes a black_box wrapping of the AMD optimization.
"""

from __future__ import division
import sys

import numpy
from mpi4py import MPI

from openmdao.api import Component, Problem, Group
from openmdao.core.petsc_impl import PetscImpl

from MAUD.core import Framework, Assembly, IndVar, MultiPtVar
from MAUD.driver_pyoptsparse import *
from MAUD.solvers import *

from Allocation.allocation_amiego import Allocation, add_quantities_alloc, load_params
from init_func import *
from MissionAnalysis.RMTS import setup_drag_rmts
from MissionAnalysis.mission import Mission, add_quantities_mission
from sumad import *


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


class AMDOptimization(Component):
    """ Simple Component wrapper that can execute the AMD model.

        Args
        ----
        fw : MAUD framework object
            The MAUD model, completely loaded but not initialized yet.
    """

    def __init__(self, fw):
        """ Create AMDOptimization instance."""
        super(AMDOptimization, self).__init__()

        self.fw = fw

        # TODO: Create vars for openmdao

    def solve_nonlinear(self, params, unknowns, resids):
        """ Pulls integer params from vector, then runs the fw model.
        """
        fw = self.fw

        # TODO: We need to pull integer design variables from params and
        # assign them into fw
        # alloc['flt_day'].value = val

	# Need to reinitialize driver with the new values each time.
        driver = DriverPyOptSparse()#options={'Verify level':3})
        fw.compute()
        fw.init_driver(driver)

        # Run
        fw.run()

        # TODO: We need to save off the Objective and Constraints, and continuous desvars


filename = 'output%03i.out'%MPI.COMM_WORLD.rank
if MPI.COMM_WORLD.rank == 0:
    filename = 'output.out'
redirectIO(open(filename, 'w'))

r2d = 180.0 / numpy.pi

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

interp, yt = setup_drag_rmts(A_list0, M_list0)

nTwist = 6
nShape = 72

DVGeo, DVCon = init_func3(nTwist)

aero_groups = []
for ipt in xrange(npt):
    ap = init_func1('fc%02i'%ipt, A_list0[ipt], M_list0[ipt])

    sys_group = Assembly('fc%02i'%ipt,
                         input_maps={
            'twist': None,
            'shape': None,
            },
                         output_maps={},
                         subsystems=[
                             IndVar('alpha', value=A_list0[ipt]),
                             SysAeroSolver('sys_aero', ap=ap,
                                           nTwist=nTwist, nShape=nShape,
                                           DVGeo=DVGeo,
                                           init_func=init_func2),
                         ])
    aero_groups.append(sys_group)

sys_aero_groups = Assembly('sys_aero_groups', subsystems=aero_groups,
                           proc_split=[])
sys_aero_groups.nonlinear_solver = NLNsolverJC(ilimit=1)
sys_aero_groups.linear_solver = LINsolverJC(ilimit=1)


problem = 'problem_128rt_4ac_1new.py'

path_file = open('./packages_path.txt', 'r')
packages_path = path_file.readlines()[0][:-1]
ac_path = packages_path + '/MissionAnalysis/inputs/'
problem_path = packages_path + '/Allocation/inputs/'
problemfile = problem_path + problem

data = {'numpy': numpy}
execfile(problemfile, data)
rt_data = data['rt_data']
ac_data = data['ac_data']
misc_data = data['misc_data']

num_rt = rt_data['number']
num_ext_ac = len(ac_data['existing_ac'])
num_new_ac = len(ac_data['new_ac'])
num_ac = num_ext_ac + num_new_ac

list_ac_params, list_mission_params = load_params(ac_path, rt_data, ac_data, misc_data)


alloc = Allocation('sys_alloc', ac_path=ac_path, rt_data=rt_data,
                   ac_data=ac_data, misc_data=misc_data,
                   interp=interp, yt=yt, num_hi=npt)


top = Assembly('sys_top', subsystems=[
    IndVar('twist', value=0*numpy.ones(nTwist)),
    IndVar('shape', value=0*numpy.ones(nShape)),
    SysDVCon('sys_dv_con', nTwist=nTwist, nShape=nShape,
             DVGeo=DVGeo, DVCon=DVCon),
    sys_aero_groups,
    MultiPtVar('sys_multipt_lift', npt=npt,
               in_name='fc%02i.cl', out_name='CL_hifi'),
    MultiPtVar('sys_multipt_drag', npt=npt,
               in_name='fc%02i.cd', out_name='CD_hifi'),
    alloc,
])

fw = Framework()
fw.init_systems(top)

fw.add_quantity('input', 'twist', indices=range(nTwist),
                lower=-10, upper=10, scale=1.0)
fw.add_quantity('input', 'shape', indices=range(nShape),
                lower=-0.5, upper=0.5, scale=10.0)
fw.add_quantity('output', 'vol_con', indices=[0],
                lower=1.0, upper=3.0, group='g:pax_con')
fw.add_quantity('output', 'thk_con', indices=range(100),
                lower=1.0, upper=3.0, group='g:thk_con')

for imsn in xrange(num_rt * num_new_ac):
    prefix = 'sys_msn%i.' % (imsn)
    num_cp = alloc[prefix[:-1]].kwargs['mission_params']['num_cp']
    num_pt = alloc[prefix[:-1]].kwargs['mission_params']['num_pt']

#    alloc[prefix[:-1]]['h_cp'].value = h_cp_list[imsn]
    alloc[prefix[:-1]]['h_cp'].value = numpy.loadtxt('msn_profiles/msn_%i.dat'%imsn)

    # Mission design variables get added here.
    add_quantities_mission(fw, prefix, num_cp, num_pt)

flt_day_init = ac_data['flt_day'].flatten(order='C')
pax_flt_init = ac_data['pax_flt'].flatten(order='C')

for ind in xrange(len(pax_flt_init)):
    if flt_day_init[ind] != 0:
        pax_flt_init[ind] /= flt_day_init[ind]

#alloc['flt_day'].value = flt_day_init.astype(float)
alloc['pax_flt'].value = pax_flt_init.astype(float)

add_quantities_alloc(fw)

# Final setup stuff
fw.init_vectors()
fw.top.set_print(False)

# Build OpenMDAO Model 

top = Problem(impl=PetscImpl())
top.root = root = Group()
root.add('amd', AMDOptimization(fw))

top.setup()

top.run()

print("Complete")

