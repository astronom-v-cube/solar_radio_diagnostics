from params import (Lparms, Nf, NSteps, ParmLocal, Rparms, freqs,
                    indexes_of_recoverable_parameters, recoverable_parameters)
from utils import Calc_I
from astropy.io import fits
import numpy as np


RL_reference = Calc_I(freqs, recoverable_parameters, indexes_of_recoverable_parameters, ParmLocal, Lparms, Rparms, NSteps, Nf)
data = np.array(RL_reference)
hdu = fits.PrimaryHDU(data)
hdu.writeto('image.fits', overwrite=True)
fits.info('image.fits')


