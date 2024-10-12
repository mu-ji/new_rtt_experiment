How to decide the number of packet used for one distance estimation?

The resolution is 9/n where n is the number of packet. Usually the larger n is the heigher resolution is.
But with larger n, the energy consumption and cost time will become larger. If the target is moving, the longer time will cause less position accuracy.

Assume the speed of target is v, the measurement time is n*tau where tau is the interval between two packet. 
To make a resonable position estimation, the moving distance of the target should smaller than the resolution, i.e.
n*tau*v <= 9/n i.e. v*n^2*tau <= 9 This equation limited the upper bound of n