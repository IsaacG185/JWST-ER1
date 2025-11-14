import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt

def dir(filter):
    return "~/Documents/JWST_Project/MAST_2025-11-13T1653/JWST/jw01727-o140_t104_nircam_clear-" + filter + "/"  # directory containing your files

filters = ["F115W", "F150W", "F277W", "F444W"]
cat_files = {
    "F115W": dir("f115w") + "jw01727-o140_t104_nircam_clear-f115w_cat.ecsv",
    "F150W": dir("f150w") + "jw01727-o140_t104_nircam_clear-f150w_cat.ecsv",
    "F277W": dir("f277w") + "jw01727-o140_t104_nircam_clear-f277w_cat.ecsv",
    "F444W": dir("f444w") + "jw01727-o140_t104_nircam_clear-f444w_cat.ecsv",
}

target_label = 66   

fluxes = []
errors = []
eff_lambda_micron = []

for filt in filters:
    cat = Table.read(cat_files[filt], format="ascii.ecsv")
    row = cat[cat["label"] == target_label][0]  # select your source

    flux = row["aper_total_flux"]  # adjust column name as needed
    err  = row["aper_total_flux_err"]

    fluxes.append(flux)
    errors.append(err)

    # Approximate effective wavelengths for NIRCam filters (in microns)
    if   filt == "F115W": eff_lambda_micron.append(1.15)
    elif filt == "F150W": eff_lambda_micron.append(1.50)
    elif filt == "F277W": eff_lambda_micron.append(2.77)
    elif filt == "F444W": eff_lambda_micron.append(4.44)

fluxes = np.array(fluxes)
errors = np.array(errors)
eff_lambda_micron = np.array(eff_lambda_micron)

plt.errorbar(eff_lambda_micron, fluxes, yerr=errors, fmt="o")
plt.xscale("log")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Flux (same units as catalog)")
plt.title("Einstein Ring SED from JWST/NIRCam")
plt.show()
