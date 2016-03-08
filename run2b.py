from __future__ import division
import numpy
import os

from MAUD.core import Framework, Assembly, IndVar
from MAUD.solvers import *

from MissionAnalysis.mission import Mission, add_quantities_mission
from Allocation.functionals import SysProfit, SysPaxCon, SysAcCon
from Allocation.allocation import Allocation, add_quantities_alloc, load_params

from MAUD.driver_pyoptsparse import *


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

from mpi4py import MPI
comm = MPI.COMM_WORLD

for irt in xrange(num_rt):
    mission_params = list_mission_params[irt]
    for iac in xrange(num_new_ac):
        ac_params = list_ac_params[iac]
        ac_name = ac_data['new_ac'][iac]
        imsn = irt + iac * num_rt

        if comm.rank == imsn:
            msn = Mission('mission', index=0, nproc=1,
                          ac_params=ac_params,
                          mission_params=mission_params)
    
            top = Assembly('top', subsystems=[
                    IndVar('pax_flt', value=ac_data['capacity', ac_name]),
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
            driver = DriverPyOptSparse(options={'Print file': 'MSN_%i_%i_print.out'%(iac,irt)},
                                       hist_file='hist_%i_%i.hst'%(iac,irt)) #options={'Verify level':3})
            fw.init_driver(driver)
            fw.top.set_print(False)
            fw.run()
            h_cp_local = msn['h_cp'].value

h_cp_list = comm.allgather(h_cp_local)

alloc = Allocation('allocation', ac_path=ac_path, rt_data=rt_data,
                   ac_data=ac_data, misc_data=misc_data)

fw = Framework()
fw.init_systems(alloc)

for imsn in xrange(len(alloc['sys_msns'].global_subsystems)):
    prefix = 'sys_msn%i.' % (imsn)
    num_cp = alloc[prefix[:-1]].kwargs['mission_params']['num_cp']
    num_pt = alloc[prefix[:-1]].kwargs['mission_params']['num_pt']

    alloc[prefix[:-1]]['h_cp'].value = h_cp_list[imsn]

    add_quantities_mission(fw, prefix, num_cp, num_pt)

flt_day_init = ac_data['flt_day'].flatten(order='C')
pax_flt_init = ac_data['pax_flt'].flatten(order='C')

for ind in xrange(len(pax_flt_init)):
    if flt_day_init[ind] != 0:
        pax_flt_init[ind] /= flt_day_init[ind]

alloc['flt_day'].value = flt_day_init.astype(float)
alloc['pax_flt'].value = pax_flt_init.astype(float)

add_quantities_alloc(fw)

driver = DriverPyOptSparse()#options={'Verify level':3})

fw.init_vectors()
fw.compute()
fw.init_driver(driver)
fw.top.set_print(False)
#    fw.top.set_print(True)
fw.run()
