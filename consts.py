import numpy as np

# Ground truth for the calibration
EXTRINSICS_R = np.asarray([[7.533745e-03, -9.999714e-01, -6.166020e-04], [1.480249e-02, 7.280733e-04, -9.998902e-01], [9.998621e-01, 7.523790e-03, 1.480755e-02]])
EXTRINSICS_T = np.atleast_2d([-4.069766e-03, -7.631618e-02, -2.717806e-01]).T

EXTRINSICS = np.block([[EXTRINSICS_R, EXTRINSICS_T], [0, 0, 0, 1]])

INSTRINSICS = np.reshape([7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00 ,7.215377e+02, 1.728540e+02, 0.000000e+00, 0.000000e+00, 1], (3, 3))