# GSPs-in-GPSes
GPS output enhancement with the use of Gaussian Stochastic Processes

In GPS_supervised_smoothing.py, there are functions for improving the output of GPS devices, based on Supervised Learning modelling. 

The input should be previous GPS positions (Latitude-Longitude) in corresponding times and the results are predictions of position for future close times. 

The Gaussian Stochastic Process modelling in based on Radial Basis function kernel plus White Noise, so the previous positions are filtered to drop likely noise and also predictions can happen for future momments under the assumption of smooth transitioning of GSP.
