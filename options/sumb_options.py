AEROSOLVER = SUMB

aeroOptions = {
    # Common Parameters
    'gridFile':gridFile,
    'outputDirectory':outputDirectory,
    
    # Physics Parameters
    'equationType':args_mode,
    
    # Common Parameters
    'CFL':CFL,
    'CFLCoarse':CFL,
    'MGCycle':MGCYCLE,
    'MGStartLevel':MGSTART,
    'nCyclesCoarse':1,#500,
    'nCycles' :1,#nCycles,
    'nsubiterturb':3,
    'useNKSolver':useNK,
    'miniterationnum':1,#50,
    
    # Convergence Parameters
    'L2Convergence':1e-1,#6,
    'L2ConvergenceCoarse':1e-1,#2,
    
    # Adjoint Parameters
    'adjointL2Convergence':1e-1,#7,
    'ADPC':True,
    'adjointMaxIter': 500,
    'adjointSubspaceSize':150, 
    'ILUFill':2,
    'ASMOverlap':1,
    'outerPreconIts':3,
}

meshOptions = {
    'gridFile':gridFile,
    'warpType':'algebraic',
}
