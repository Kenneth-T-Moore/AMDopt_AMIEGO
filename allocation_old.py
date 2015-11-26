from __future__ import division
import numpy
import os

from MAUD.core import Framework, Assembly, IndVar
from MAUD.solvers import *

from MissionAnalysis.mission import *
from Allocation.functionals import SysProfit, SysPaxCon, SysAcCon

try:
    from MAUD.driver_pyoptsparse import *
except:
    pass



def add_quantities_alloc(fw):
    alloc = fw.top

    num_rt, num_ac = alloc.num_rt, alloc.num_ac
    pax_upper, demand, avail = get_bounds(alloc)
    
    fw.add_quantity('output', 'profit_1e6_d', group='g:profit')
    fw.add_quantity('input', 'pax_flt', indices=range(num_rt*num_ac),
                    lower=0, upper=pax_upper)
    fw.add_quantity('input', 'flt_day', indices=range(num_rt*num_ac),
                    lower=0, upper=10)
    fw.add_quantity('output', 'pax_con', indices=range(num_rt),
                    lower=0.1*demand, upper=2*demand, group='g:pax_con')
    fw.add_quantity('output', 'ac_con', indices=range(num_ac),
                    upper=avail, group='g:ac_con')

def get_bounds(alloc):
    rt_data, ac_data, misc_data = alloc.rt_data, alloc.ac_data, alloc.misc_data

    num_rt = rt_data['number']
    num_ext_ac = len(ac_data['existing_ac'])
    num_new_ac = len(ac_data['new_ac'])
    num_ac = num_ext_ac + num_new_ac

    pax_upper = numpy.zeros(num_rt * num_ac)
    upper = pax_upper.reshape((num_rt, num_ac), order='F')
    for iac in xrange(num_ac):
        if iac < num_ext_ac:
            ac_name = ac_data['existing_ac'][iac]
        else:
            inac = iac - num_ext_ac
            ac_name = ac_data['new_ac'][inac]
        for irt in xrange(num_rt):
            upper[irt, iac] = ac_data['capacity', ac_name]

    demand = numpy.zeros(num_rt)
    for irt in xrange(num_rt):
        demand[irt] = rt_data['demand'][irt]

    avail = numpy.zeros(num_ac)
    for iac in xrange(num_ac):
        if iac < num_ext_ac:
            ac_name = ac_data['existing_ac'][iac]
        else:
            inac = iac - num_ext_ac
            ac_name = ac_data['new_ac'][inac]
        avail[iac] = 2 * 12 * ac_data['number', ac_name]

    return pax_upper, demand, avail

def load_params(ac_path, rt_data, ac_data, misc_data):
    num_rt = rt_data['number']
    num_ext_ac = len(ac_data['existing_ac'])
    num_new_ac = len(ac_data['new_ac'])
    num_ac = num_ext_ac + num_new_ac

    list_ac_params = []
    for ac_name in ac_data['new_ac']:
        list_ac_params.append(get_ac_params(ac_path + ac_name))

    list_mission_params = []
    for irt in xrange(num_rt):
        rng_nmi = rt_data['range'][irt]
        mission_params = {
            'num_cp': int(min(50, rng_nmi/50. + 5)),
            'num_pt': int(min(50, rng_nmi/50. + 5))*5,
            'range_1e3_km': rng_nmi * 1.852 / 1e3,
            'alt_init_km': 10.,
            }
        jac_B, jac_dBdx = get_Bspline_props(mission_params)
        mission_params['jac_B'] = jac_B
        mission_params['jac_dBdx'] = jac_dBdx
        list_mission_params.append(mission_params)

    return list_ac_params, list_mission_params


class Allocation(Assembly):

    def initialize(self):
        ac_path = self.kwargs['ac_path']
        rt_data = self.kwargs['rt_data']
        ac_data = self.kwargs['ac_data']
        misc_data = self.kwargs['misc_data']
        self.rt_data, self.ac_data, self.misc_data = rt_data, ac_data, misc_data
        
        num_rt = rt_data['number']
        num_ext_ac = len(ac_data['existing_ac'])
        num_new_ac = len(ac_data['new_ac'])
        num_ac = num_ext_ac + num_new_ac
        self.num_rt, self.num_ac = num_rt, num_ac

        list_ac_params, list_mission_params = load_params(ac_path, rt_data, ac_data, misc_data)

        msns = []
        for irt in xrange(num_rt):
            mission_params = list_mission_params[irt]
            mission_params['interp'] = self.kwargs['interp']
            mission_params['num_hi'] = self.kwargs['num_hi']
            for iac in xrange(num_new_ac):
                ac_params = list_ac_params[iac]
                
                index = irt + iac * num_rt
                index2 = index + num_ext_ac * num_rt

                msn = Mission('sys_msn%i' % (index), index=index2,
                              input_maps={
                                  'pax_flt': None,
                                  'CD_hifi': None,
                                  'CL_hifi': None,
                              },
                              output_maps={},
                              ac_params=ac_params,
                              mission_params=mission_params)
                msns.append(msn)

        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            nproc = comm.size
        except:
            nproc = 1

        sys_inputs = Assembly('sys_inputs', subsystems=[
            IndVar('pax_flt', value=10 * numpy.ones(num_rt*num_ac)),
            IndVar('flt_day', value=0.1 * numpy.ones(num_rt*num_ac)),
        ])
        if nproc > 1:
            sys_msns = Assembly('sys_msns', subsystems=msns,
                                proc_split=[])
            sys_msns.nonlinear_solver = NLNsolverJC(ilimit=1)
            sys_msns.linear_solver = LINsolverJC(ilimit=1)
        else:
            sys_msns = Assembly('sys_msns', subsystems=msns)
            sys_msns.nonlinear_solver = NLNsolverGS(ilimit=1)
            sys_msns.linear_solver = LINsolverGS(ilimit=1)
            
        sys_funcs = Assembly('sys_funcs', subsystems=[
            SysProfit('sys_profit', misc_data=misc_data,
                      ac_data=ac_data, rt_data=rt_data),
            SysPaxCon('sys_paxcon', num_rt=num_rt, num_ac=num_ac),
            SysAcCon('sys_accon', misc_data=misc_data,
                     ac_data=ac_data, rt_data=rt_data),
        ])

        self.add_subsystem(sys_inputs)
        self.add_subsystem(sys_msns)
        self.add_subsystem(sys_funcs)
