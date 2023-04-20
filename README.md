# GSPs-in-GPSes
GPS output enhancement with the use of Gaussian Stochastic Processes

In GPS_supervised_smoothing.py, there are functions for improving the output of GPS devices, based on Supervised Learning modelling. 

The input should be previous GPS positions (Latitude-Longitude) in corresponding times and the results are predictions of position for future close times. 

The Gaussian Stochastic Process modelling in based on Radial Basis function kernel plus White Noise, so the previous positions are filtered to drop likely noise and also predictions can happen for future momments under the assumption of smooth transitioning of GSP.

Ideally the user should work with the library as follows,
---
import GPS_supervised_smoothing

t = * list with previous times \
la = * list with previous latituded \
lon = * list with previous longitudes \
t_+ = * list with next time of interest \

reshape_kernel(la, lon) 

times = t.append(t_+)  \
times = reshape_time(initial_times = times) \
GSPpred( times[:-1], lon, la, times[-1] )  \
---

Comments
- You don't need to reshape_kernel often, once in the beginning would be ok
- You don't need to include all the previous values in GSPpred, 5 at least is recommended, more than 20 would be ideal
- Each time you use GSPpred, model is updated, so it keeps in account all previous inputs, even if not included
