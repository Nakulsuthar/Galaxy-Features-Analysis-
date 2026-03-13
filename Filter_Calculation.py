import numpy as np


wavelength, transmission = np.loadtxt(
    "HST_ACS_WFC.F814W (1).dat",
    unpack=True
)


integral = np.trapz(transmission, wavelength)

print(f"Integral is: {integral:.1f} Å")

from scipy.integrate import cumulative_trapezoid

cumulative_area = cumulative_trapezoid(
    transmission,
    wavelength,
    initial=0
)

half_area = 0.5 * cumulative_area[-1]

lambda_half = np.interp(
    half_area,
    cumulative_area,
    wavelength
)

print(lambda_half)