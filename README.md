# GSPs-in-GPSes
GPS output enhancement with the use of Gaussian Stochastic Processes

In kalman_filter.py, there is the classic approach of Kalman Filter in the form of python functions.

In GPS_supervised_smoothing.py, there is GSP class for improving the output of GPS devices, based on Supervised Learning modelling. 

The input should be previous GPS positions (Latitude-Longitude) in corresponding times and the results are predictions of position for future close times. Also, the previous course can be returned corrected as model performance indice.

The Gaussian Stochastic Process modelling is based on Radial Basis function kernel plus White Noise, so the previous positions are filtered to drop likely noise and also predictions can happen for future momments under the assumption of smooth transitioning of GSP.

Ideally the functions should work as follows,
```python
import GPS_supervised_smoothing

t =   [10,11,22]                # list with previous times
la =  [42.1,42.2,42.3]          # list with previous latitudes
lon = [-12.1,-12.2,-12.3]       # list with previous longitudes
t_new = t[-1] + 1               # list with next time of interest

model = GPS_supervised_smoothing.GSP()
model.update(lon=lon, lat=la, time=t)
model.corrected_course(t)
model.predict_next(t_new)
```

Comments:
- You don't need to include all the previous values as sample, 5 at least is recommended, more than 20 would be ideal.
- Each time model is updated, all previous inputs are reseted.
