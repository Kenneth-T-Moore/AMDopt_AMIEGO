import os
import numpy
from mpi4py import MPI
from baseclasses import *
#from tripan import TRIPAN
from sumb import SUMB
from pywarp import *
from pygeo import *
from pyspline import *
from multipoint import *
from pyoptsparse import Optimization, OPT


args_mode = 'euler'
args_output = 'res2/'
args_shape = 0
args_opt = 'snopt'
args_optOptions = {}

outputDirectory = args_output

eulerFile = 'input/euler_grid.cgns'
ransFile = 'input/rans_grid_l2.cgns'
triFile = 'input/mdo_tutorial.tripan'
wakeFile = 'input/mdo_tutorial.wake'
edgeFile = 'input/mdo_tutorial.edge'
FFDFile = 'input/mdo_tutorial_ffd.fmt'

if args_mode == 'tripan':
    execfile('options/tripan_options.py')
elif args_mode == 'euler':
    execfile('options/euler_options.py')
    execfile('options/sumb_options.py')
elif args_mode == 'rans':
    execfile('options/rans_options.py')
    execfile('options/sumb_options.py')

def init_func1(name, A, M):
    ap = AeroProblem(name=name, mach=M, altitude=10000,
                     areaRef=45.5, alpha=A, chordRef=3.25,
                     evalFuncs=['cl', 'cd'])
    ap.addDV('alpha', value=A, lower=0, upper=10.0, scale=0.1)
    return ap

def init_func2(comm, DVGeo):
    # Create solver
    CFDSolver = AEROSOLVER(options=aeroOptions, comm=comm)
    CFDSolver.setDVGeo(DVGeo)

    if args_mode == 'euler' or args_mode == 'rans':
        mesh = MBMesh(options=meshOptions, comm=comm)
        CFDSolver.setMesh(mesh)

    return CFDSolver

def init_func3(nTwist):
    # Call common geometry setup
    DVGeo = DVGeometry(FFDFile)

    # Setup curves for ref_axis
    x = [5.0/4.0, 1.5/4.0 + 7.5]
    y = [0, 0]
    z = [0, 14]

    tmp = pySpline.Curve(x=x, y=y, z=z, k=2)
    X = tmp(numpy.linspace(0, 1, nTwist))
    c1 = pySpline.Curve(X=X, k=2)
    DVGeo.addRefAxis('wing', c1)

    def twist(val, geo):
        # Set all the twist values
        for i in xrange(nTwist):
            geo.rot_z['wing'].coef[i] = val[i]

    DVGeo.addGeoDVGlobal('twist', 0*numpy.ones(nTwist), twist,
                         lower=-10, upper=10, scale=1.0)

    DVGeo.addGeoDVLocal('shape', lower=-0.5, upper=0.5, axis='y', scale=10.0)




    CFDSolver = init_func2(MPI.COMM_WORLD, DVGeo)




    DVCon = DVConstraints()
    DVCon.setDVGeo(DVGeo)

    # Only SUmb has the getTriangulatedSurface Function
    DVCon.setSurface(CFDSolver.getTriangulatedMeshSurface())

    # Le/Te constraints
#    DVCon.addLeTeConstraints(0, 'iLow')
#    DVCon.addLeTeConstraints(0, 'iHigh')
 
    # Volume constraints
    leList = [[0.1, 0, 0.001], [0.1+7.5, 0, 14]]
    teList = [[4.2, 0, 0.001], [8.5, 0, 14]]
    DVCon.addVolumeConstraint(leList, teList, 20, 20, lower=1.0)

    # Thickness constraints
    DVCon.addThicknessConstraints2D(leList, teList, 10, 10, lower=1.0)

    return DVGeo, DVCon
