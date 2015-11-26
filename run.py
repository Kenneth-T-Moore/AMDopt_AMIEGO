from __future__ import division
import numpy

from MAUD.core import Framework, Assembly, IndVar
from sumad import *
from MAUD.driver_pyoptsparse import *
from init_func import *

nTwist = 6
npt = 2

M_list = [.78, 0.80, 0.82]
CL_list = [0.5, 0.5, 0.5]
   

DVGeo = init_func3(nTwist)

groups = []
for ipt in xrange(npt):
    ap = init_func1('fc%02i'%ipt, M_list[ipt])

    sys_group = Assembly('fc%02i'%ipt,
                         input_maps={'twist': None},
                         output_maps={},
                         subsystems=[
                             IndVar('alpha', value=1.5),
                             SysAeroSolver('sys_aero', ap=ap,
                                           nTwist=nTwist,
                                           DVGeo=DVGeo,
                                           init_func=init_func2),
                             SysLiftCon('sys_lift_con',
                                        cl0=CL_list[ipt]),
                         ])
    groups.append(sys_group)


top = Assembly('sys_top', subsystems=[
    IndVar('twist', value=0*numpy.ones(nTwist)),
    Assembly('sys_fc_group', subsystems=groups,
             proc_split=[],
         ),
    SysObj('sys_obj', npt=npt)
])

fw = Framework()
fw.add_quantity('input', 'twist', indices=range(nTwist),
                lower=-10, upper=10, scale=1.0)
for ipt in xrange(npt):
    fw.add_quantity('input', 'fc%02i.alpha'%ipt,
                    lower=0, upper=10.0, scale=0.1)
fw.add_quantity('output', 'obj', group='par')
for ipt in xrange(npt):
    fw.add_quantity('output', 'fc%02i.cl_con'%ipt,
                    lower=0., upper=0., group='par')

fw.init_systems(top)
fw.init_vectors()
fw.compute()

driver = DriverPyOptSparse()#options={'Verify level':3})
fw.init_driver(driver)
fw.run()

exit()

for key in top.uvec.IDs:
    if top.iproc == 0:
        try:
            print key, top.uvec[key]
        except:
            pass

if top.iproc == 0:
    print
    print '-----------------------'
    print

for key in top.uvec.IDs:
    if top.iproc == 0:
        try:
            print key, top.rvec[key]
        except:
            pass
