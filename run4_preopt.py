""" Generate preoptimization runs from an initial Halton distribution on flights/day.

NOTE: This version uses integer values for the Halton sampling.

Notes: Twist scale 1000.0, Shape scale 1000.0, RMTS drag offset .015

Only run points where we have enough planes to be feasible.
"""

from __future__ import division
import os
from glob import glob
from sqlitedict import SqliteDict

import numpy

from mpi4py import MPI

from MAUD.core import Framework, Assembly, IndVar, MultiPtVar
from MAUD.solvers import *
from sumad import *
from MAUD.driver_pyoptsparse import *
from init_func import *
from Allocation.allocation_amiego import Allocation, add_quantities_alloc, load_params
from MissionAnalysis.RMTS15 import setup_drag_rmts
from MissionAnalysis.mission import Mission, add_quantities_mission

import sys

rank = MPI.COMM_WORLD.rank

do_missions = True

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




r2d = 180.0 / numpy.pi



A_list0 = []
M_list0 = []

A_list0.extend([-1.] * 6)
M_list0.extend([0.60, 0.70, 0.73, 0.76, 0.78, 0.80])

A_list0.extend([1.] * 6)
M_list0.extend([0.60, 0.70, 0.73, 0.76, 0.78, 0.80])

A_list0.extend([3] * 4)
M_list0.extend([0.60, 0.70, 0.74, 0.77])


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


#problem = 'problem_32rt_3ac_1new.py'
#problem = 'problem_4rt_3ac_1new.py'
problem = 'problem_128rt_4ac_1new.py'
#problem = 'problem_3rt_2ac.py'

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




'''
comm = MPI.COMM_WORLD

for irt in xrange(num_rt):
    mission_params = list_mission_params[irt]
    mission_params['interp'] = interp
    mission_params['num_hi'] = npt
    mission_params['yt'] = None
    for iac in xrange(num_new_ac):
        ac_params = list_ac_params[iac]
        ac_name = ac_data['new_ac'][iac]
        imsn = irt + iac * num_rt

        if comm.rank == imsn:
            msn = Mission('mission', index=0, nproc=1,
                          ac_params=ac_params,
                          mission_params=mission_params,
                          interp=interp, yt=None)

            top = Assembly('top', subsystems=[
                    IndVar('pax_flt', value=ac_data['capacity', ac_name]),
                    IndVar('CL_hifi', value=numpy.loadtxt('surr_CL_hifi.dat')),
                    IndVar('CD_hifi', value=numpy.loadtxt('surr_CD_hifi.dat')),
                    msn,
            ])

            num_cp = mission_params['num_cp']
            num_pt = mission_params['num_pt']

            subcomm = comm.Split(imsn)

            fw = Framework()
            add_quantities_mission(fw, '', num_cp, num_pt, True)
            fw.init_systems(top, subcomm)
            fw.init_vectors()
            fw.compute()
            driver = DriverPyOptSparse(options={
                    'Print file': 'MSN_%i_%i_print.out'%(iac,irt),
                    'Summary file': 'MSN_%i_%i_summary.out'%(iac,irt),
                    },
                                       hist_file='hist_%i_%i.hst'%(iac,irt)) #options={'Verify level':3})
            fw.init_driver(driver)
            fw.top.set_print(False)
            fw.run()
            h_cp_local = msn['h_cp'].value

#if comm.rank >= num_rt * num_new_ac:
#    subcomm = comm.Split(num_rt * num_new_ac)
#    h_cp_local = None

h_cp_list = comm.allgather(h_cp_local)
'''



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
                lower=-10, upper=10, scale=1.)
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

    #alloc[prefix[:-1]]['h_cp'].value = numpy.loadtxt('msn_profiles/msn_%i.dat'%imsn)

    #in_name = '../AMDopt/msn_profiles/hist_0_%i.hst'
    #db = SqliteDict(in_name%imsn)
    #alloc[prefix[:-1]]['M0'].value = db[db['last']]['xuser']['M0']
    alloc[prefix[:-1]]['M0'].value = .82
    if do_missions and imsn < 8:
        add_quantities_mission(fw, prefix, num_cp, num_pt)

    # HACK: try something
    #fw.add_quantity('input', prefix+'M0', indices=[0],
    #                lower=0.82, upper=0.82)

flt_day_init = ac_data['flt_day'].flatten(order='C')
pax_flt_init = ac_data['pax_flt'].flatten(order='C')

for ind in xrange(len(pax_flt_init)):
    if flt_day_init[ind] != 0:
        pax_flt_init[ind] /= flt_day_init[ind]

alloc['flt_day'].value = flt_day_init.astype(float)
alloc['pax_flt'].value = pax_flt_init.astype(float)

add_quantities_alloc(fw)



driver = DriverPyOptSparse(options={})
#options={'Verify level':3})

fw.init_vectors()
#fw.top.print_var_sizes()
#exit()
fw.compute()
fw.init_driver(driver)
fw.top.set_print(False)


import cPickle as pickle
from Allocation.allocation2 import get_bounds

pax_upper, demand, avail = get_bounds(alloc)

num_rt = 128
num_ac = 5

raw = numpy.loadtxt('halton19_int.dat')
flt_day_init = numpy.zeros((num_ac, num_rt))

turnaround = 1.0
time_hr = numpy.zeros((num_ac, num_rt))
maint = numpy.zeros((num_ac, num_rt))

for iac in xrange(num_ac):
    if iac < num_ext_ac:
        ac_name = ac_data['existing_ac'][iac]
    maint[iac, :] = ac_data['MH', ac_name]
    for irt in xrange(num_rt):
        if iac < num_ext_ac:
            time_hr[iac, irt] = ac_data['block time', ac_name][irt]
time_hr[4, :] = time_hr[2, :]

print("Begin")

for irun in range(raw.shape[0]):

    # Skip whatever we already have done:
    #if irun < 25:
    #    continue
    #if irun in [0, 4, 6, 20, 26, 27, 28, 29, 30]:
    #    continue

    flt_day_init[1, 1:5] = raw[irun, :4]
    flt_day_init[3, 1:8] = raw[irun, 4:11]
    flt_day_init[4, :8] = raw[irun, 11:19]

    for iac in xrange(num_ac):
        for irt in xrange(num_rt):
            if time_hr[iac, irt] > 1e14:
                flt_day_init[iac, irt] = 0.

    data = flt_day_init * (time_hr * (1 + maint) + turnaround)
    ac_con = numpy.sum(data, axis=1)
    if numpy.all(ac_con < avail):

        alloc['flt_day'].value = flt_day_init.flatten()

        #options={'Verify level':3}
        options={'Print file' : 'SNOPT_print_%03i' % irun,
                 'Major feasibility tolerance' : 1e-5,
                 'Major optimality tolerance' : 1e-3}
        driver = DriverPyOptSparse(options=options)
        fw.init_driver(driver)
        fw.top.set_print(False)
        fw.run()

        if rank == 0:
            db = SqliteDict('mrun/hist.hst')
            dvs_dict = db[db['last']]['xuser']
            funcs_dict = db[db['last']]['funcs']
            db.close()

            pickle.dump(dvs_dict, open( 'pre_data/dvs_%03i.pkl' % irun, "wb" ) )
            pickle.dump(funcs_dict, open( 'pre_data/funcs_%03i.pkl' % irun, "wb" ) )

            files = glob('mrun/*.hst')
            for file in files:
                os.remove(file)
            files = glob('res/*')
            for file in files:
                os.remove(file)
            #files = glob('output*')
            #for file in files:
            #os.remove(file)

    else:
        print("Skipping Case:", irun)
        print("/n")
        print("avail", avail)
        print("ac_con", ac_con)

        # Print some values to use in place of the bad case.
        con_val = ac_con - avail
        pax[iac, irt] = (seat_cap[iac] * demand[irt] )/ sum_over_iac(seat[iac] * \
                         flight_day_init[iac,irt])
        #This is a fair	assumption given our cost also stays constant in the objective equ.	
        #So we should be able to calculate profit (obj) quite accurately.
        profit[iac, irt] = pax[iac, irt] * flight_day_init[iac,ir]*price[irt] - cost[iac, irt]*flight_day_init[iac, irt]
        obj_val	= sum(sum(profit)) 
        print("USE THESE")
        print('Cons', con_val)
        print('Obj', obj_val)

print("End!")
