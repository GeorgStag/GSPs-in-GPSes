########################################################################
########################################################################
########################################################################
### libraries


import numpy as np
from colorama import Fore
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF


########################################################################
########################################################################
########################################################################
### model initilizations


kernel_coor = RBF(1.0) + WhiteKernel(noise_level=0.4, noise_level_bounds=(0.01, 0.01))

gpr_lon = GaussianProcessRegressor(kernel=kernel_coor, random_state=0)

gpr_la = GaussianProcessRegressor(kernel=kernel_coor, random_state=0)


########################################################################
########################################################################
########################################################################
### function to reshape time


def reshape_time(initial_times):

    if type(initial_times) != list:
        print(Fore.RED + 'Warning: Time must imputted as 1D list!!!')

    ind = []
    for i in initial_times:
        ind.append( float(i) )
    ind = np.array(ind)

    if min(ind) <= 0:
        ind = ind - min(ind) + 1

    if max(ind) > 10:
        k = 0
        for i in ind:
            ind[k] = np.log(ind[k])
            k = k + 1

    if max(ind) >= 10:
        ind = ind / ( max(ind) - 5 )

    return ind


########################################################################
########################################################################
########################################################################
### function to update kernels


def reshape_kernel(logitude_sample, latitude_sample):

    print(Fore.RED + 'Warning: models will be resetted from any previous training!!!\n...')
    global gpr_lon, gpr_la

    meds = [np.median(logitude_sample), np.median(latitude_sample)]
    logitude_sample = np.array(logitude_sample) - meds[0]
    latitude_sample = np.array(latitude_sample) - meds[1]

    meds = [np.median(logitude_sample), np.median(latitude_sample)]
    maxs = [np.quantile(logitude_sample, 0.9), np.quantile(latitude_sample, 0.9)]
    mins = [np.quantile(logitude_sample, 0.25), np.quantile(latitude_sample, 0.25)]

    kernel_coor = RBF(1.0) + WhiteKernel(noise_level=meds[0], noise_level_bounds=(mins[0], maxs[0]))
    gpr_lon = GaussianProcessRegressor(kernel=kernel_coor, random_state=0)

    kernel_coor = RBF(1.0) + WhiteKernel(noise_level=meds[1], noise_level_bounds=(mins[1], maxs[1]))
    gpr_la = GaussianProcessRegressor(kernel=kernel_coor, random_state=0)

    return print(Fore.GREEN + '200: Success, kernel updated')


########################################################################
########################################################################
########################################################################
### function to predict new positions


def GSPpred( initial_times, logitude_sample, latitude_sample, future_time):

    global gpr_lon, gpr_la

    print(Fore.BLUE + 'Suggestions: If results are not good please think of reshaping time and updating kernels!!!')

    if type(logitude_sample) != list:
        print(Fore.RED + 'Warning: Variables must imputted as 1D lists!!!')

    ind = []
    for i in initial_times:
        ind.append( float(i) )
    ind = np.array(ind).reshape(-1, 1)

    meds = [np.median(logitude_sample), np.median(latitude_sample)]

    logitude_sample = np.array(logitude_sample) - meds[0]
    latitude_sample = np.array(latitude_sample) - meds[1]

    ind_pre = np.array(future_time[0]).reshape(1, -1)

    gpr_lon.fit(ind, logitude_sample)
    outp_lon = gpr_lon.predict(ind_pre)

    gpr_la.fit(ind, latitude_sample)
    outp_la = gpr_la.predict(ind_pre)

    return [xnew, outp_lon[0] + med[0], outp_la[0] + med[1]]
