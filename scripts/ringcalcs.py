import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from astropy import constants as const

# --- Inputs ---
z_l = 3.147        # lens redshift
z_s = 6.220        # source redshift
theta_E_arcsec = 0.60   # your measured Einstein radius

# Distances in meters
D_l  = cosmo.angular_diameter_distance(z_l).to(u.m)
D_s  = cosmo.angular_diameter_distance(z_s).to(u.m)
D_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).to(u.m)

print(f"D_l  = {D_l.to(u.Mpc):.3f}")
print(f"D_s  = {D_s.to(u.Mpc):.3f}")
print(f"D_ls = {D_ls.to(u.Mpc):.3f}")

# Einstein radius: radians → *dimensionless float*
theta_E = (theta_E_arcsec * u.arcsec).to(u.rad).value  # no units now

# Mass inside Einstein radius
M_E = (const.c**2 / (4 * const.G) * (D_l * D_s / D_ls) * theta_E**2).to(u.Msun)
print(f"\nM(<θ_E) = {M_E:.3e}")

# Physical Einstein radius at the lens plane
R_E = (theta_E * D_l).to(u.kpc)
print(f"R_E (physical) = {R_E:.2f}")
