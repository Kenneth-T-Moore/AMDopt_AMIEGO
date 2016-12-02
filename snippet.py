# Testing out some new code for pre-checking the AMD constraints

import numpy as np


problem = 'problem_128rt_4ac_1new.py'
problem_path = '../Allocation/inputs/'
problemfile = problem_path + problem
data = {'numpy': np}
execfile(problemfile, data)

ac = data['ac_data']['existing_ac']
ac.append('B777')

seat_cap = np.zeros((5, ))
for j in range(5):
    seat_cap[j] = data['ac_data'][('capacity', ac[j])]

price = np.zeros((5, 8))
cost = np.zeros((5, 8))
for iac in range(5):
    for irt in range(8):
        price[iac, irt] = data['ac_data'][('ticket price', ac[iac])][irt]
        fuel = data['ac_data'][('fuel', ac[iac])][irt]
        base_cost = data['ac_data'][('flight cost no fuel', ac[iac])][irt]
        cost[iac, irt] = base_cost + fuel*data['misc_data']['cost/fuel']
        pass

demand = data['rt_data']['demand']

flt_day_raw = [ 6.,  6.,  6.,  4.,  0.,  0.,  6.,  1.,  6.,  0.,  0.,  0.,  0.,  4.,  5.,  3.,  6.,  3., 2.]
flt_day = np.zeros((5, 8))
flt_day[1, 1:5] = flt_day_raw[:4]
flt_day[3, 1:] = flt_day_raw[4:11]
flt_day[4, :] = flt_day_raw[11:]

#pax[iac, irt] = min(seat_cap[iac], (seat_cap[iac] * demand[irt])/sum_over_iac(seat[iac]*flight_day_init[iac,irt]))
pax = np.zeros((5, 8))
for iac in range(5):
    for irt in range(8):
        calc_pax = (seat_cap[iac]*demand[irt])/sum(seat_cap[iac]*flt_day)
        pax[iac, irt] = min(seat_cap[iac], calc_pax[irt])

#profit[iac, irt] = pax[iac, irt]*flight_day_init[iac, ir]*price[irt] - cost[iac, irt]*flight_day_init[iac, irt]
profit = np.zeros((5, 8))
for iac in range(5):
    for irt in range(8):
        profit[iac, irt] = pax[iac, irt]*flt_day[iac, irt]*price[iac, irt] - cost[iac, irt]*flt_day[iac, irt]

# Zero out rt/ac combos that don't fly
profit[0, :] = 0
profit[2, :] = 0
profit[1, (0, 5, 6, 7)] = 0
profit[3, 0] = 0

profit = np.array([-sum(sum(profit))/1.e6])
print('Bad Case Profit', profit)

#ac_con = 2400*np.ones((5, ))
ac_con = np.zeros((5, 8))
#data = flt_day * (time_hr * (1 + maint) + turnaround)
#ac_con[:, 0] = numpy.sum(data, axis=0)
for iac in [1, 3, 4]:
    MH = data['ac_data'][('MH', ac[iac])]
    time = data['ac_data'][('block time', ac[iac])]
    for irt in range(8):
        ac_con[iac, irt] += flt_day[iac, irt] * (time[irt] * (1 + MH) + 1.0)

# filter out impossible runs
#ac_con[1, (0, 5, 6, 7)] = 0
#ac_con[3, 0] = 0
ac_con = np.sum(ac_con, axis=1)
print('Bad Case ac_con', ac_con)

#pax_con = 2.0*np.array([   10.,   108.,  1396.,  3145.,   995.,  4067.,   639.,   321.])
#pax_con = np.zeros((8, ))
pax_con = sum(pax*flt_day)
print('Bad Case pax_con', pax_con)

pass
