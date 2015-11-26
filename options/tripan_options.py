AEROSOLVER = TRIPAN

aeroOptions = {'outputDirectory':outputDirectory,
               'writeSolution':True,
               'dragMethod':'total',
               'useSymmetry':True,
               'nWakeCells':1,
               'printIterations':False,
               'numberSolutions':True,
               'wakeLength':455.0,
               'tripanFile':triFile,
               'wakeFile':wakeFile,
               'edgeFileList':[edgeFile]}
