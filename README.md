# GSPs-in-GPSes
GPS output enhancement with the use of Gaussian Stochastic Processes

In GPS_supervised_smoothing.py, there are functions for improving the output of GPS devices, based on Supervised Learning modelling. 

The input should be previous GPS positions (Latitude-Longitude) in corresponding times and the results are predictions of position for future close times. 

The Gaussian Stochastic Process modelling in based on Radial Basis function kernel plus White Noise, so the previous positions are filtered to drop likely noise and also predictions can happen for future momments under the assumption of smooth transitioning of GSP.

Ideally the user should work with the library as follows,

import GPS_supervised_smoothing

t = * list with previous times
la = * list with previous latituded
lon = * list with previous longitudes
t_+ = * list with next time of interest

reshape_kernel(la, lon) ## you don't need to update kernels often 

times = t.append(t_+) 
times = reshape_time(initial_times = times)
GSPpred( times[:-1], lon, la, times[-1] ) ## you don't need all the previous values, 5 previous values would be good, 20 or more would be ideal, 
                                          ## each time you use the function the model is updated
