from __future__ import division
import numpy

from MAUD.core import ExplicitComponent, Assembly, IndVar


class SysAeroSolver(ExplicitComponent):

    def initialize(self):
        nTwist = self.kwargs['nTwist']
        nShape = self.kwargs['nShape']

        self.add_input('twist', range(nTwist))
        self.add_input('shape', range(nShape))
        self.add_input('alpha', [0])
        self.add_output('cl', 1.0)
        self.add_output('cd', 1.0)

        if self.comm is not None:
            self.kwargs['CFDSolver'] = self.kwargs['init_func'](self.comm, self.kwargs['DVGeo'])

    def oper_execute(self):
        print("SUMAD start.")

        ap = self.kwargs['ap']
        DVGeo = self.kwargs['DVGeo']
        CFDSolver = self.kwargs['CFDSolver']

        dv_dict = {}
        dv_dict['twist'] = self.pvec['twist'][:, 0]
        dv_dict['shape'] = self.pvec['shape'][:, 0]
        dv_dict[ap.DVNames['alpha']] = self.pvec['alpha'][0, 0]

        DVGeo.setDesignVars(dv_dict)
        ap.setDesignVars(dv_dict)

	print("Hey Justin CFD", ap.mach, ap.altitude)
        func_dict = {}
        CFDSolver(ap)
        CFDSolver.evalFunctions(ap, func_dict)
        #CFDSolver.checkSolutionFailure(ap, func_dict)

        if 'fail' in func_dict:
            if func_dict['fail']:
                print("Ken, Fail in Func Dict")
                self.rvec.oper_set_const(1.0)

        # If it fails this way, reset flow and try again.
        #if ap.solveFailed and not ap.fatalFail:
        if numpy.isnan(CFDSolver.getResNorms()[2]) or (CFDSolver.getResNorms()[2] > 1.0e-7 and not ap.fatalFail):
            func_dict = {}
            print("Ken, Retrying", ap.solveFailed, ap.fatalFail)
            print(CFDSolver.getResNorms())
            CFDSolver.resetFlow(ap)
            CFDSolver(ap)
            CFDSolver.evalFunctions(ap, func_dict)
        
        #if ap.fatalFail or ap.solveFailed:
        if numpy.isnan(CFDSolver.getResNorms()[2]) or CFDSolver.getResNorms()[2] > 1.0e-7 or ap.fatalFail:
            self.rvec.oper_set_const(1.0)
            CFDSolver.resetFlow(ap)
            print("Ken, Failed a Second Time")

        for name in ['cl', 'cd']:
            self.uvec[name][0, 0] = func_dict[ap[name]]

    def oper_jacobians(self):
        ap = self.kwargs['ap']
        DVGeo = self.kwargs['DVGeo']
        CFDSolver = self.kwargs['CFDSolver']

        #if (not ap.fatalFail) and (not ap.solveFailed):
        if not (CFDSolver.getResNorms()[2] > 1.0e-7 or ap.fatalFail):
            sens_dict = {}
            CFDSolver.evalFunctionsSens(ap, sens_dict)

            for vout in ['cl', 'cd']:
                vin = 'twist'
                self.jacobians[vout, vin] = sens_dict[ap[vout]][vin]
                vin = 'shape'
                self.jacobians[vout, vin] = sens_dict[ap[vout]][vin]
                vin = 'alpha'
                self.jacobians[vout, vin] = sens_dict[ap[vout]][ap.DVNames[vin]]



class SysDVCon(ExplicitComponent):

    def initialize(self):
        nTwist = self.kwargs['nTwist']
        nShape = self.kwargs['nShape']

        DVCon = self.kwargs['DVCon']
        jacs = DVCon.linearCon['DVCon1_lete_constraint_0'].jac
        self.lete_jac0 = jacs['shape']
        jacs = DVCon.linearCon['DVCon1_lete_constraint_1'].jac
        self.lete_jac1 = jacs['shape']

        self.add_input('twist', range(nTwist))
        self.add_input('shape', range(nShape))
        self.add_output('vol_con')
        self.add_output('thk_con', numpy.zeros(100))
        self.add_output('lete0_con', numpy.zeros(6))
        self.add_output('lete1_con', numpy.zeros(6))

    def oper_execute(self):
        DVGeo = self.kwargs['DVGeo']
        DVCon = self.kwargs['DVCon']

        dv_dict = {}
        dv_dict['twist'] = self.pvec['twist'][:, 0]
        dv_dict['shape'] = self.pvec['shape'][:, 0]

        DVGeo.setDesignVars(dv_dict)

        func_dict = {}
        DVCon.evalFunctions(func_dict)

        shape = self.pvec['shape'][:, 0]
        self.uvec['vol_con'][0, 0] = func_dict['DVCon1_volume_constraint_0']
        self.uvec['thk_con'][:, 0] = func_dict['DVCon1_thickness_constraints_0']
        self.uvec['lete0_con'][:, 0] = self.lete_jac0.dot(shape)
        self.uvec['lete1_con'][:, 0] = self.lete_jac1.dot(shape)

    def oper_jacobians(self):
        DVCon = self.kwargs['DVCon']

        sens_dict = {}
        DVCon.evalFunctionsSens(sens_dict)

        for vin in ['twist', 'shape']:
            self.jacobians['vol_con', vin] = sens_dict['DVCon1_volume_constraint_0'][vin]
            self.jacobians['thk_con', vin] = sens_dict['DVCon1_thickness_constraints_0'][vin]

        self.jacobians['lete0_con', 'shape'] = self.lete_jac0
        self.jacobians['lete1_con', 'shape'] = self.lete_jac1
        



class SysObj(ExplicitComponent):

    def initialize(self):
        npt = self.kwargs['npt']

        for ipt in xrange(npt):
            self.add_input('fc%02i.cd'%ipt, [0])

        self.add_output('obj', 1.0)

    def oper_execute(self):
        npt = self.kwargs['npt']

        obj = 0
        for ipt in xrange(npt):
            obj += self.pvec['fc%02i.cd'%ipt][0, 0]
        self.uvec['obj'][0, 0] = obj

    def oper_jacobians(self):
        npt = self.kwargs['npt']

        for ipt in xrange(npt):
            self.jacobians['obj', 'fc%02i.cd'%ipt] = 1.0


class SysLiftCon(ExplicitComponent):

    def initialize(self):
        self.add_input('cl', [0])
        self.add_output('cl_con', 1.0)

    def oper_execute(self):
        cl0 = self.kwargs['cl0']

        self.uvec['cl_con'][0, 0] = self.pvec['cl'][0, 0] - cl0

    def oper_jacobians(self):
        self.jacobians['cl_con', 'cl'] = 1.0
