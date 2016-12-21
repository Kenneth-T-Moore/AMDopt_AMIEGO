AEROSOLVER = SUMB

aeroOptions = {
    # Common Parameters
    'gridFile':gridFile,
    'outputDirectory':outputDirectory,
    'isoSurface':{'shock':1.0, 'vx':-0.001},
    
    # Physics Parameters
    'equationType':args_mode,
    
    # Common Parameters
    'CFL':CFL,
    'CFLCoarse':CFL,
    'MGCycle':'3v',
    'MGStartLevel':1,
    'nCyclesCoarse':500,
    'nCycles' :3000,
    'nsubiterturb':3,
    'useNKSolver':useNK,
    'miniterationnum':50,
    
    # Convergence Parameters
    'L2Convergence':1e-12,
    'L2ConvergenceCoarse':1e-2,
    'nkadpc': False, 

    # Adjoint Parameters
    'adjointL2Convergence':1e-10,
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
