""" OpenMDAO model that builds wraps and executes a black_box wrapping of the
AMD optimization.
"""

from __future__ import division
import pickle
from six.moves import range
import sys

import numpy as np
from mpi4py import MPI
from sqlitedict import SqliteDict

from openmdao.api import Component, Problem, Group, Driver, IndepVarComp
from openmdao.core.petsc_impl import PetscImpl
from openmdao.drivers.amiego_driver import AMIEGO_driver

from MAUD.core import Framework, Assembly, IndVar, MultiPtVar
from MAUD.driver_pyoptsparse import *
from MAUD.solvers import *

from Allocation.allocation_amiego import Allocation, add_quantities_alloc, load_params
from init_func import *
from MissionAnalysis.RMTS15 import setup_drag_rmts
from MissionAnalysis.mission import Mission, add_quantities_mission
from sumad import *


"""
This shows the design variables that should be culled because they assign a
flt/day to aircraft that cannot physically fly the route.

[1:5 10:]

data['ac_data'][('block time', 'B738')][:8]
array([  1.00000000e+15,   5.40044600e+00,   4.73839400e+00,
         5.17117800e+00,   5.15223600e+00,   1.00000000e+15,
         1.00000000e+15,   1.00000000e+15])
data['ac_data'][('block time', 'B747')][:8]
array([  1.00000000e+15,   5.04604000e+00,   4.43854000e+00,
         4.83672903e+00,   4.81867194e+00,   6.54317778e+00,
         6.78287778e+00,   6.35066444e+00])
data['ac_data'][('block time', 'B777')][:8]
array([ 14.8867    ,   5.200842  ,   4.57873708,   4.98679504,
         4.96874848,   6.742045  ,   6.98413   ,   6.53992   ])
"""

class AMDOptimization(Component):
    """ Simple Component wrapper that can execute the AMD model.

    Args
    ----
    fw : MAUD framework object
        The MAUD model, completely loaded but not initialized yet.

    alloc: Assembly
        Pointer to the allocation assembly. Passed in because I am not sure MAUD
        has a way to get it.

    init_func : dict
        Initial values of obj/constraints - needed just for sizing.
    """
    def __init__(self, fw, alloc, top, init_func, init_dv):
        """ Create AMDOptimization instance."""
        super(AMDOptimization, self).__init__()

        self.fw = fw
        self.alloc = alloc
        self.top = top
        self.init_dv = init_dv

        self.iter_count = 0

        # Integer input that AMIEGO will set.

        self.add_param('flt_day', np.zeros((19, ), dtype=np.int))

        # Continuous desvars are just outputs.
        # Note, it is not necessary for AMIEGO to know about all of these. Just for
        # simplicity's sake, AMIEGO will track progress on passengers per flight,
        # but twist, shape, and the cp an M0 for each mission will be ignored here,
        # but saved off in the history file.

        #n_i = len(alloc['pax_flt'].value.shape)
        #self.add_output('pax_flt', np.zeros((n_i, )))

        # Objective Output
        self.add_output('profit_1e6_d', 0.0)

        # We just need the constraints impacted by the integer vars.
        self.add_output('ac_con', np.zeros((3, )))
        self.add_output('pax_con_upper', np.zeros((8, )))
        #self.add_output('pax_con_lower', np.zeros((8, )))
        #self.add_output('thk_con', np.zeros((100, )))
        #self.add_output('vol_con', 0.0)

        # The gamma constraint is sized differently for each mission. Far easier to
        # just read the sizes from the initial saved case.
        #for j in range(8):
            #root = 'sys_msn%d' % j

            #for var in ['gamma', 'Tmax', 'Tmin']:
                #name_i = root + '.' + var
                #name_o = root + ':' + var
                #size = len(init_func[name_i])
                #self.add_output(name_o, np.zeros((size,)))

    def solve_nonlinear(self, params, unknowns, resids):
        """ Pulls integer params from vector, then runs the fw model.
        """
        fw = self.fw
        alloc = self.alloc
        top = self.top
        init_dv = self.init_dv

        # Pull integer design variables from params and assign them into fw
        flt_day_init = numpy.zeros((num_ac, num_rt))
        raw = params['flt_day']
        flt_day_init[1, 1:5] = raw[:4]
        flt_day_init[3, 1:8] = raw[4:11]
        flt_day_init[4, :8] = raw[11:19]
        alloc['flt_day'].value = flt_day_init.flatten()

        # Load initial real design values from one of the preopts
        # Using first point now, which is best.
        # Set initial conditions from best preopt
        alloc['pax_flt'].value = init_dv['pax_flt'].flatten()
        top['shape'].value = init_dv['shape'].flatten()
        top['twist'].value = init_dv['twist'].flatten()
        for j in range(8):
            root = 'sys_msn%d.' % j
            for var in ['M0', 'h_cp']:
                name = root + var
                alloc[prefix[:-1]][var].value = init_dv[name].flatten()

        # Reinitialize driver with the new values each time.
        options={'Print file' : 'AMIEGO_%03i' % self.iter_count,
                 'Major feasibility tolerance' : 1e-6,
                 'Major optimality tolerance' : 5e-5}
        driver = DriverPyOptSparse(options=options)
        fw.init_driver(driver)
        fw.top.set_print(False)

        # Run
        fw.run()
        self.iter_count += 1

        # Load in optimum from SNOPT history
        db = SqliteDict('mrun/hist.hst')
        dvs_dict = db[db['last']]['xuser']
        funcs_dict = db[db['last']]['funcs']
        db.close()

        # Objective
        unknowns['profit_1e6_d'] = funcs_dict['profit_1e6_d']

        # Constraints
        unknowns['ac_con'] = funcs_dict['ac_con'][[1, 3, 4]]
        unknowns['pax_con_upper'] = funcs_dict['pax_con'][:8]
        #unknowns['pax_con_lower'] = funcs_dict['pax_con'][:8]
        #unknowns['thk_con'] = funcs_dict['thk_con']
        #unknowns['thk_con'] = funcs_dict['thk_con']
        #unknowns['vol_con'] = funcs_dict['vol_con']

        # Save out the case
        if not MPI or self.comm.rank == 0:

            pickle.dump(dvs_dict, open( 'post_data/dvs_%03i.pkl' % self.iter_count, "wb" ) )
            pickle.dump(funcs_dict, open( 'post_data/funcs_%03i.pkl' % self.iter_count, "wb" ) )


class AMDDriver(Driver):
    """ A run-once driver that can report its error state just like
    pyoptsparse."""

    def __init__(self, fw):
        """ Create AMDDriver instance."""
        super(AMDDriver, self).__init__()
        self.fw = fw

    def run(self, problem):
        """ Runs the driver. This function should be overridden when inheriting.

        Args
        ----
        problem : `Problem`
            Our parent `Problem`.
        """
        super(AMDDriver, self).run(problem)

        # Let AMIEGO know whether AMD optimization passed or failed.
        self.success = self.fw.driver.success


#filename = 'output%03i.out'%MPI.COMM_WORLD.rank
#if MPI.COMM_WORLD.rank == 0:
    #filename = 'output.out'
#redirectIO(open(filename, 'w'))

r2d = 180.0 / np.pi

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

# Load pickles for initial sampling
dv_samp = pickle.load( open( "../good_preopts/dv_samp.pkl", "rb" ) )
obj_samp = pickle.load( open( "../good_preopts/obj_samp.pkl", "rb" ) )
con_samp = pickle.load( open( "../good_preopts/con_samp.pkl", "rb" ) )
dv_samp_bad = pickle.load( open( "../good_preopts/dv_samp_w_bad.pkl", "rb" ) )
#obj_samp = pickle.load( open( "../good_preopts/obj_samp_w_bad.pkl", "rb" ) )
#con_samp = pickle.load( open( "../good_preopts/con_samp_w_bad.pkl", "rb" ) )

#-------------------------------------
# Warmer Start from Initial Conditions
#-------------------------------------
init_dv = pickle.load( open( "../good_int_preopts/dvs_045.pkl", "rb" ) )
init_func = pickle.load( open( "../good_int_preopts/funcs_045.pkl", "rb" ) )

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

data = {'numpy': np}
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

dvcon = SysDVCon('sys_dv_con', nTwist=nTwist, nShape=nShape,
             DVGeo=DVGeo, DVCon=DVCon)

top = Assembly('sys_top', subsystems=[
    IndVar('twist', value=init_dv['twist']),
    IndVar('shape', value=init_dv['shape']),
    dvcon,
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
                lower=-10, upper=10, scale=5000.0)
fw.add_quantity('input', 'shape', indices=range(nShape),
                lower=-0.5, upper=0.5, scale=5000.0)
fw.add_quantity('output', 'vol_con', indices=[0],
                lower=1.0, upper=3.0, group='g:pax_con')
fw.add_quantity('output', 'thk_con', indices=range(100),
                lower=1.0, upper=3.0, group='g:thk_con')

for imsn in xrange(num_rt * num_new_ac):
    prefix = 'sys_msn%i.' % (imsn)
    num_cp = alloc[prefix[:-1]].kwargs['mission_params']['num_cp']
    num_pt = alloc[prefix[:-1]].kwargs['mission_params']['num_pt']

    alloc[prefix[:-1]]['h_cp'].value = np.loadtxt('msn_profiles/msn_%i.dat'%imsn)

    # Mission design variables get added here.
    # Optimizer only considers the first 8 routes.
    if imsn < 8:
        #print('ken', imsn, num_cp)
        add_quantities_mission(fw, prefix, num_cp, num_pt)

flt_day_init = ac_data['flt_day'].flatten(order='C')
pax_flt_init = ac_data['pax_flt'].flatten(order='C')

for ind in xrange(len(pax_flt_init)):
    if flt_day_init[ind] != 0:
        pax_flt_init[ind] /= flt_day_init[ind]

alloc['pax_flt'].value = pax_flt_init.astype(float)

add_quantities_alloc(fw)

# Final MAUD setup stuff
fw.init_vectors()
fw.compute()
fw.top.set_print(False)

# Set initial conditions from best preopt
alloc['pax_flt'].value = init_dv['pax_flt']
top['shape'].value = init_dv['shape']
top['twist'].value = init_dv['twist']
for j in range(8):
    root = 'sys_msn%d.' % j
    for var in ['M0', 'h_cp']:
        name = root + var
        print(j, alloc[prefix[:-1]][var].value.shape, init_dv[name].shape)
        alloc[prefix[:-1]][var].value = init_dv[name]

#----------------------
# Build OpenMDAO Model
#----------------------

prob = Problem(impl=PetscImpl)
prob.root = root = Group()
root.add('p1', IndepVarComp('flt_day', np.zeros((19, ), dtype=np.int)), promotes=['*'])
root.add('amd', AMDOptimization(fw, alloc, top, init_func, init_dv), promotes=['*'])

prob.driver = AMIEGO_driver()
prob.driver.cont_opt = AMDDriver(fw)

# To save time
prob.driver.minlp.options['atol'] = 0.1
prob.driver.minlp.options['local_search'] = True
prob.driver.minlp.options['penalty_factor'] = 0.5
prob.driver.minlp.options['maxiter'] = 100000
prob.driver.minlp.options['maxiter_ubd'] = 10000
prob.driver.options['ei_tol_rel'] = 0.00001

demand = np.array([   10.,   108.,  1396.,  3145.,   995.,  4067.,   639.,   321.])

prob.driver.add_desvar('flt_day', lower=0, upper=6)
prob.driver.add_objective('profit_1e6_d')
prob.driver.add_constraint('ac_con', upper=2400.0)
prob.driver.add_constraint('pax_con_upper', upper=2.0*demand)
#prob.driver.add_constraint('pax_con_lower', lower=0.0)

# Only create surrogates for the constraints that are used. They blow up
# otherwise.
con_samp_pax = []
for samp in con_samp['pax_con']:
    con_samp_pax.append(samp[:8])

con_samp_ac = []
for samp in con_samp['ac_con']:
    con_samp_ac.append(samp[[1, 3, 4]])

reduced_con_samp = {}
reduced_con_samp['ac_con'] = con_samp_ac
reduced_con_samp['pax_con_upper'] = con_samp_pax
#reduced_con_samp['pax_con_lower'] = con_samp_pax

# Reduce DVs too to get rid of impossible cases.
reduced_dv_samp = {}
reduced_dv_samp['flt_day'] = []
for samp in dv_samp['flt_day']:
    reduced_dv_samp['flt_day'].append(samp[[1, 2, 3, 4,
                                            9, 10, 11, 12, 13, 14, 15,
                                            16, 17, 18, 19, 20, 21, 22, 23]])

# Reduce bad DVs too to get rid of impossible cases.
reduced_dv_samp_bad = {}
reduced_dv_samp_bad['flt_day'] = []
for samp in dv_samp_bad['flt_day'][28:]:
    reduced_dv_samp_bad['flt_day'].append(samp[[1, 2, 3, 4,
                                            9, 10, 11, 12, 13, 14, 15,
                                            16, 17, 18, 19, 20, 21, 22, 23]])
prob.driver.sampling = reduced_dv_samp
prob.driver.obj_sampling = obj_samp
prob.driver.con_sampling = reduced_con_samp

# Remove these for now. If anything, try adding just the int failures later on.
#prob.driver.minlp.bad_samples

prob.setup()

prob.run()

print("Complete")

