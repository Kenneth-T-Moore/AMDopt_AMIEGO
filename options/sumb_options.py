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
    'nCyclesCoarse':500,
    'nCycles' :1000,
    'nsubiterturb':3,
    'useNKSolver':useNK,
    'miniterationnum':50,
    
    # Convergence Parameters
    'L2Convergence':1e-6,
    'L2ConvergenceCoarse':1e-2,
    
    # Adjoint Parameters
    'adjointL2Convergence':1e-7,
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
