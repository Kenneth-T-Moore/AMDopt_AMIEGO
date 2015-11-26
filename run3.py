from __future__ import division
import numpy

from MAUD.core import Framework, Assembly, IndVar, MultiPtVar
from MAUD.solvers import *
from sumad import *
from MAUD.driver_pyoptsparse import *
from init_func import *
from Allocation.allocation2 import Allocation, add_quantities_alloc, load_params
from MissionAnalysis.RMTS import setup_drag_rmts
from MissionAnalysis.mission import add_quantities_mission





nA = 2
nM = 2

A_list0 = numpy.linspace(-0.02, 0.10, nA)
M_list0 = numpy.linspace(0.67, 0.82, nM)

A_list = []
M_list = []
for iA in xrange(nA):
    for iM in xrange(nM):
        A_list.append(A_list0[iA])
        M_list.append(M_list0[iM])
npt = nA * nM

interp, yt = setup_drag_rmts(A_list0, M_list0)






nTwist = 6

DVGeo = init_func3(nTwist)

aero_groups = []
for ipt in xrange(npt):
    ap = init_func1('fc%02i'%ipt, M_list[ipt])

    sys_group = Assembly('fc%02i'%ipt,
                         input_maps={'twist': None},
                         output_maps={},
                         subsystems=[
                             IndVar('alpha', value=A_list[ipt]),
                             SysAeroSolver('sys_aero', ap=ap,
                                           nTwist=nTwist,
                                           DVGeo=DVGeo,
                                           init_func=init_func2),
                         ])
    aero_groups.append(sys_group)

sys_aero_groups = Assembly('sys_aero_groups', subsystems=aero_groups,
                           proc_split=[])
sys_aero_groups.nonlinear_solver = NLNsolverJC(ilimit=1)
sys_aero_groups.linear_solver = LINsolverJC(ilimit=1)



problem = 'problem_4rt_3ac_1new.py'
#problem = 'problem_128rt_4ac_1new.py'
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

alloc = Allocation('sys_alloc', ac_path=ac_path, rt_data=rt_data,
                   ac_data=ac_data, misc_data=misc_data,
                   interp=interp, yt=yt, num_hi=npt)




top = Assembly('sys_top', subsystems=[
    IndVar('twist', value=0*numpy.ones(nTwist)),
#    sys_aero_groups,
#    MultiPtVar('sys_multipt_lift', npt=npt,
#               in_name='fc%02i.cl', out_name='CL_hifi'),
#    MultiPtVar('sys_multipt_drag', npt=npt,
#               in_name='fc%02i.cd', out_name='CD_hifi'),
    alloc,
])

fw = Framework()
fw.init_systems(top)

fw.add_quantity('input', 'twist', indices=range(nTwist),
                lower=-10, upper=10, scale=1.0)

for imsn in xrange(npt):
    prefix = 'sys_msn%i.' % (imsn)
    num_cp = alloc[prefix[:-1]].kwargs['mission_params']['num_cp']
    num_pt = alloc[prefix[:-1]].kwargs['mission_params']['num_pt']

    #alloc[prefix[:-1]]['h_cp'].value = h_cp_list[imsn]

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
#fw.top.print_var_sizes()
#exit()
fw.compute()
fw.init_driver(driver)
fw.top.set_print(False)
#fw.top.set_print(True)
fw.run()
