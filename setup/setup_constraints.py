# ======================================================================
#         DVConstraint Setup
# ====================================================================== 
DVCon = DVConstraints()
DVCon.setDVGeo(DVGeo)

if args_shape:
    # Only SUmb has the getTriangulatedSurface Function
    DVCon.setSurface(CFDSolver.getTriangulatedMeshSurface())

    # Le/Te constraints
    DVCon.addLeTeConstraints(0, 'iLow')
    DVCon.addLeTeConstraints(0, 'iHigh')
 
    # Volume constraints
    leList = [[0.1, 0, 0.001], [0.1+7.5, 0, 14]]
    teList = [[4.2, 0, 0.001], [8.5, 0, 14]]
    DVCon.addVolumeConstraint(leList, teList, 20, 20, lower=1.0)

    # Thickness constraints
    DVCon.addThicknessConstraints2D(leList, teList, 10, 10, lower=1.0)

    if comm.rank == 0:
        fileName = os.path.join(args.output, 'constraints.dat')
        DVCon.writeTecplot(fileName)
