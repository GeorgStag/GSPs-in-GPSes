

import numpy as np
from colorama import Fore
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
import warnings
warnings.filterwarnings("ignore")


class GSP:

  def __init__(self):
    self.kernel_coor_lon = RBF(1.0) + WhiteKernel( noise_level=0.4, noise_level_bounds=(0.01, 0.01))
    self.kernel_coor_lat = RBF(1.0) + WhiteKernel( noise_level=0.4, noise_level_bounds=(0.01, 0.01))
    self.gpr_lon = GaussianProcessRegressor( kernel = self.kernel_coor_lon, random_state=0)
    self.gpr_la = GaussianProcessRegressor( kernel = self.kernel_coor_lat, random_state=0)
    self.lon = []
    self.lat = []
    self.centers = [0,0]
    self.times = []
    self.r_const = [0,6,False]

  def scale_times(self, times):
    reshape_const = self.r_const
    ind = []
    for i in times:
      ind.append( float(i) )
    ind = np.array(ind)
    if min(ind) <= 0:
      reshape_const[0] = min(ind)
      ind = ind - min(ind) + 1
    else:
      reshape_const[0] = 0
    if max(ind) > 10:
      reshape_const[2] = True
      k = 0
      for i in ind:
          ind[k] = np.log(ind[k])
          k = k + 1
    else:
      reshape_const[2] = False
    if max(ind) >= 10:
      reshape_const[1] = max(ind)
      ind = ind / ( max(ind) - 5 )
    else:
      reshape_const[1] = 6
    print(Fore.GREEN + "Scaling Successful; Reshape constants updated!!!")
    return ind

  def update_kernels(self, lon, lat):
    print(Fore.RED + 'Warning: models will be resetted from any previous training!!!\n...')
    self.centers = [np.median(lon), np.median(lat)]
    logitude_sample = np.array(lon) - self.centers[0]
    latitude_sample = np.array(lat) - self.centers[1]
    meds = [np.median(logitude_sample), np.median(latitude_sample)]
    maxs = [np.quantile(logitude_sample, 0.9), np.quantile(latitude_sample, 0.9)]
    mins = [np.quantile(logitude_sample, 0.25), np.quantile(latitude_sample, 0.25)]
    self.kernel_coor_lon = RBF(1.0) + WhiteKernel(noise_level=meds[0], noise_level_bounds=(mins[0], maxs[0]))
    self.gpr_lon = GaussianProcessRegressor( kernel=self.kernel_coor_lon, random_state=0)
    self.kernel_coor_lat = RBF(1.0) + WhiteKernel(noise_level=meds[1], noise_level_bounds=(mins[1], maxs[1]))
    self.gpr_la = GaussianProcessRegressor( kernel=self.kernel_coor_lat, random_state=0)
    return print(Fore.GREEN + 'Success; Kernels updated!!!')

  def update(self, lon, lat, time, scale_time = True, update_kernels = True):
    if type(lat) != list or type(lon) != list or type(time) != list :
      print(Fore.RED + 'Warning: Longitude, Latitude and time must imputted as 1D lists respectively!!!')
    if update_kernels : self.update_kernels(lon,lat)
    if scale_time : time = self.scale_times(time)
    time = np.array(time).reshape(1, -1)
    self.gpr_lon.fit(time.reshape(-1, 1), lon - self.centers[0])
    self.gpr_la.fit(time.reshape(-1, 1), lat - self.centers[1])
    return print(Fore.GREEN + 'Success; GSP trained!!!')

  def predict_next(self, new_time):
    reshape_const = self.r_const
    new_time = new_time - reshape_const[0]
    if reshape_const[2]: new_time = np.log(new_time)
    new_time = new_time/(reshape_const[1]-5)
    new_time = np.array(new_time).reshape( 1, -1)
    outp_lon = self.gpr_lon.predict(new_time) + self.centers[0]
    outp_la = self.gpr_la.predict(new_time) + self.centers[1]
    return [outp_lon[0], outp_la[0]]

  def corrected_course(self, time, scale_time = True):
    if scale_time : time = self.scale_times(time)
    time = np.array(time).reshape(-1, 1)
    outp_lon = self.gpr_lon.predict(time) + self.centers[0]
    outp_la = self.gpr_la.predict(time) + self.centers[1]
    return np.array([outp_lon, outp_la])
