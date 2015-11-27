from __future__ import division
import numpy

from MAUD.core import Framework, Assembly, IndVar
from MAUD.solvers import *
from sumad import *
from MAUD.driver_pyoptsparse import *
from init_func import *

nTwist = 6
nShape = 72
npt = 2

M_list = [.78, 0.80, 0.82]
A_list = [0, 3]
CL_list = [0.5, 0.5, 0.5]
   

DVGeo, DVCon = init_func3(nTwist)


'''
execfile('options/euler_options.py')
execfile('options/sumb_options.py')
from mpi4py import MPI

ap = AeroProblem(name='name', mach=0.78, altitude=10000,
                 areaRef=45.5, alpha=0, chordRef=3.25,
                 evalFuncs=['cl', 'cd'])
CFDSolver = AEROSOLVER(options=aeroOptions, comm=MPI.COMM_WORLD)
CFDSolver.setDVGeo(DVGeo)
mesh = MBMesh(options=meshOptions, comm=MPI.COMM_WORLD)
CFDSolver.setMesh(mesh)
CFDSolver(ap)
exit()
'''



groups = []
for ipt in xrange(npt):
    ap = init_func1('fc%02i'%ipt, A_list[ipt], M_list[ipt])

    sys_group = Assembly('fc%02i'%ipt,
                         input_maps={
            'twist': None,
            'shape': None,
            },
                         output_maps={},
                         subsystems=[
                             IndVar('alpha', value=1.5),
                             SysAeroSolver('sys_aero', ap=ap,
                                           nTwist=nTwist, nShape=nShape,
                                           DVGeo=DVGeo,
                                           init_func=init_func2),
                             SysLiftCon('sys_lift_con',
                                        cl0=CL_list[ipt]),
                         ])
    groups.append(sys_group)


sys_aero_groups = Assembly('sys_fc_group', subsystems=groups,
)#                           proc_split=[])
sys_aero_groups.nonlinear_solver = NLNsolverJC(ilimit=1)
sys_aero_groups.linear_solver = LINsolverJC(ilimit=1)

top = Assembly('sys_top', subsystems=[
    IndVar('twist', value=0*numpy.ones(nTwist)),
    IndVar('shape', value=0*numpy.ones(nShape)),
    SysDVCon('sys_dv_con', nTwist=nTwist, nShape=nShape,
             DVGeo=DVGeo, DVCon=DVCon),
    sys_aero_groups,
    SysObj('sys_obj', npt=npt)
])

fw = Framework()
fw.add_quantity('input', 'twist', indices=range(nTwist),
                lower=-10, upper=10, scale=1.0)
fw.add_quantity('input', 'shape', indices=range(nShape),
                lower=-0.5, upper=0.5, scale=10.0)
fw.add_quantity('output', 'vol_con', indices=[0],
                lower=1.0, upper=3.0, group='g:pax_con')
fw.add_quantity('output', 'thk_con', indices=range(100),
                lower=1.0, upper=3.0, group='g:thk_con')

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
