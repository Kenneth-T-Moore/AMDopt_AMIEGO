# ======================================================================
#         DVGeometry Setup
# ====================================================================== 
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

if True:
    DVGeo.addGeoDVLocal('shape', lower=-0.5, upper=0.5, axis='y', scale=10.0)
    


# # =====================================================
# #                Setup Design Variable Functions

# def twist(val, geo):
#     # Set all but the root twist values:
#     for i in xrange(1,len(val)+1):
#         geo.rot_z[0].coef[i] = val[i-1]

# def thickness(val, geo):
#     # Set all the thickness scaling factors
#     for i in xrange(len(val)):
#         geo.scale_y[0].coef[i] = val[i]

# def chords(val, geo):

#     s = geo.refAxis.curves[0].s
#     # Interpolate chords:
#     for i in xrange(len(s)):
#         geo.scale[0].coef[i] = (s[i]-s[0])/(s[-1]-s[0])*(val[1]-val[0]) + val[0]

# def span(val, geo):
#     C = geo.extractCoef(0)
#     s = geo.refAxis.curves[0].s

#     for i in xrange(len(C)):
#         C[i,2] = C[i,2] + s[i]*val[0]
#     geo.restoreCoef(C, 0)

# def sweep(val, geo):

#     C = geo.extractCoef(0)
#     s = geo.refAxis.curves[0].s

#     for i in xrange(len(C)):
#         C[i,0] = C[i,0] + s[i]*val[0]
#     geo.restoreCoef(C, 0)


# coef = FFD.vols[0].coef.copy()

# # First determine the reference chord lengths:                                                                                                                         
# nSec = coef.shape[2]
# sweep_ref = numpy.zeros((nSec,3))

# for k in xrange(nSec):
#     max_x = numpy.max(coef[:,:,k,0])
#     min_x = numpy.min(coef[:,:,k,0])
#     sweep_ref[k,0] = min_x + 0.25*(max_x-min_x)
#     sweep_ref[k,1] = numpy.average(coef[:,:,k,1])
#     sweep_ref[k,2] = numpy.average(coef[:,:,k,2])

# sweep_ref[0,2] = 0.0 # Precise zero at mirror plane
# c0 = pySpline.curve(X=sweep_ref,k=2)
# DVGeo = DVGeometry.DVGeometry([solver_coords], [c0], axis = ['x'],FFD=FFD,
#                               rot_type=5, names = ['solver_coords'])

# # Add the global geometric design variables
# DVGeo.addGeoDVGlobal('span',0, -5, 5, span)
# DVGeo.addGeoDVGlobal('sweep',0,-3, 3, sweep)
# DVGeo.addGeoDVGlobal('chords',numpy.ones(2), .5*numpy.ones(2), 
#                      1.5*numpy.ones(2), chords)
# DVGeo.addGeoDVGlobal('thickness', numpy.ones(nSec), .5*numpy.ones(nSec),
#                      1.5*numpy.ones(nSec), thickness)
# DVGeo.addGeoDVGlobal('twist',numpy.zeros(nSec-1),-10*numpy.ones(nSec-1),
#                      10*numpy.ones(nSec-1), twist)
