import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy.table import Table

# -----------------------------
# USER INPUTS
# -----------------------------
# Center of your Einstein ring (in degrees)
center_ra  = 150.100480
center_dec =   1.893035

# Size of region (in degrees) around center; you've already tuned this
region_width_deg = 0.001       # example (~36 arcsec); adjust as you like
half_width_deg   = region_width_deg / 2.0

# Aperture radii (in arcseconds)
# lens: circular aperture around the central galaxy
# ring: annulus between r_lens_arcsec and r_ring_arcsec
r_lens_arcsec = 0.3          # inner radius (lens)
r_ring_arcsec = 1.05          # outer radius (ring annulus)
# tweak these by eyeballing the overplotted circles

# Map filter name -> i2d FITS file path

def dir(filter):
    return "~/Documents/JWST_Project/MAST_2025-11-13T1653/JWST/jw01727-o140_t104_nircam_clear-" + filter + "/"  # directory containing your files

filter_files = {
    "F115W": dir("f115w") + "jw01727-o140_t104_nircam_clear-f115w_i2d.fits",
    "F150W": dir("f150w") + "jw01727-o140_t104_nircam_clear-f150w_i2d.fits",
    "F277W": dir("f277w") + "jw01727-o140_t104_nircam_clear-f277w_i2d.fits",
    "F444W": dir("f444w") + "jw01727-o140_t104_nircam_clear-f444w_i2d.fits",
}

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_pixscale_deg(header):
    """
    Estimate pixel scale in degrees/pixel from WCS-related keywords.
    Works for typical JWST i2d headers.
    """
    # Case 1: CDELT
    if "CDELT1" in header:
        return abs(header["CDELT1"])

    # Case 2: CD matrix
    cd11 = header.get("CD1_1", None)
    cd12 = header.get("CD1_2", None)
    if cd11 is not None and cd12 is not None:
        return np.sqrt(cd11**2 + cd12**2)

    # Case 3: PC + CDELT
    pc11 = header.get("PC1_1", None)
    pc12 = header.get("PC1_2", None)
    cdelt1 = header.get("CDELT1", None)
    if pc11 is not None and pc12 is not None and cdelt1 is not None:
        return abs(cdelt1) * np.sqrt(pc11**2 + pc12**2)

    # Fallback: approximate NIRCam pixel scale (~0.031"/pix)
    print("Warning: could not determine pixel scale from header; using ~0.031\"/pix default")
    return 0.031 / 3600.0


def aperture_flux(sub_sci, sub_err, mask, pixar_sr):
    """
    Compute total flux and error in μJy for a given mask on a cutout.

    sub_sci, sub_err: 2D arrays (MJy/sr) for SCI and ERR in the cutout
    mask: boolean 2D array (same shape)
    pixar_sr: pixel area in steradians (from header PIXAR_SR)
    """
    sci_mjysr = sub_sci[mask]
    err_mjysr = sub_err[mask]

    # Convert MJy/sr per pixel -> μJy per pixel: MJy/sr * sr * 1e6
    flux_uJy_pix = sci_mjysr * (pixar_sr * 1e6)
    err_uJy_pix  = err_mjysr  * (pixar_sr * 1e6)

    flux_tot = flux_uJy_pix.sum()
    err_tot  = np.sqrt((err_uJy_pix**2).sum())
    return flux_tot, err_tot


# -----------------------------
# MAIN: build 2x2 grid with masks
# -----------------------------
fig = plt.figure(figsize=(8, 8))

filters = ["F115W", "F150W", "F277W", "F444W"]

# Store photometry results if you want them later
lens_fluxes = {}
lens_errors = {}
ring_fluxes = {}
ring_errors = {}

for idx, filt in enumerate(filters):
    fname = filter_files[filt]
    print(f"\n=== {filt} ===")
    print(f"Loading {fname}")

    with fits.open(fname) as hdul:
        sci_hdu = hdul["SCI"] if "SCI" in hdul else hdul[1]
        sci_data = sci_hdu.data           # MJy/sr
        header = sci_hdu.header
        wcs = WCS(header)

        # ERR extension has the uncertainties
        err_data = hdul["ERR"].data       # MJy/sr
        # pixel area in steradians from header
        pixar_sr = header.get("PIXAR_SR", None)
        if pixar_sr is None:
            raise RuntimeError(f"PIXAR_SR not found in header of {fname}")

    ny, nx = sci_data.shape

    # Pixel scale in deg/pix and arcsec/pix
    pixscale_deg = get_pixscale_deg(header)
    pixscale_arcsec = pixscale_deg * 3600.0

    # Center in pixel coordinates
    x_arr, y_arr = wcs.world_to_pixel_values(center_ra, center_dec)
    x_center = float(x_arr)
    y_center = float(y_arr)

    # Region size in pixels
    half_width_pix = half_width_deg / pixscale_deg

    # Pixel bounds around the center (clipped)
    x_min = int(np.round(x_center - half_width_pix))
    x_max = int(np.round(x_center + half_width_pix))
    y_min = int(np.round(y_center - half_width_pix))
    y_max = int(np.round(y_center + half_width_pix))

    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, nx)
    y_max = min(y_max, ny)

    # Cutouts
    sub_data = sci_data[y_min:y_max, x_min:x_max]
    sub_err  = err_data[y_min:y_max, x_min:x_max]

    # WCS for the cutout
    wcs_sub = wcs.slice((slice(y_min, y_max), slice(x_min, x_max)))

    # Coordinates in cutout pixel frame
    ny_sub, nx_sub = sub_data.shape
    yy, xx = np.indices((ny_sub, nx_sub))

    # Center position in cutout coordinates
    x0 = x_center - x_min
    y0 = y_center - y_min

    # Radial distance (in pixels) from center
    r_pix = np.sqrt((xx - x0)**2 + (yy - y0)**2)

    # Convert aperture radii from arcsec to pixels
    r_lens_pix = r_lens_arcsec / pixscale_arcsec
    r_ring_pix = r_ring_arcsec / pixscale_arcsec

    # Define masks
    lens_mask      = (r_pix <= r_lens_pix)
    ring_ann_mask  = (r_pix > r_lens_pix) & (r_pix <= r_ring_pix)  # used for flux

    # Compute photometry in μJy (optional but useful)
    lens_flux, lens_err = aperture_flux(sub_data, sub_err, lens_mask,     pixar_sr)
    ring_flux, ring_err = aperture_flux(sub_data, sub_err, ring_ann_mask, pixar_sr)

    lens_fluxes[filt] = lens_flux
    lens_errors[filt] = lens_err
    ring_fluxes[filt] = ring_flux
    ring_errors[filt] = ring_err

    print(f"Lens  flux (μJy): {lens_flux:.3g} ± {lens_err:.3g}")
    print(f"Ring  flux (μJy): {ring_flux:.3g} ± {ring_err:.3g}")

#     # ---- Plotting ----
#     ax = fig.add_subplot(2, 2, idx + 1, projection=wcs_sub)

#     norm = simple_norm(sub_data, "sqrt", percent=99.0, clip=True)
#     im = ax.imshow(sub_data, origin="lower", norm=norm, cmap="gray")

#     ax.set_title(filt)
#     ax.set_xlabel("RA")
#     ax.set_ylabel("Dec")

#     # Mark center
#     ax.plot(x0, y0, marker="+", color="red", markersize=8, mew=1.5,
#             transform=ax.get_transform("pixel"))

#     ring_outer_mask = (r_pix <= r_ring_pix)   # simple disk, so contour gives 1 circle

#     # Overplot lens and ring masks as contours (0.5 separates False/True)
#     ax.contour(lens_mask,      levels=[0.5], colors="yellow",
#             linewidths=1.0, linestyles="solid",
#             transform=ax.get_transform("pixel"))

#     ax.contour(ring_outer_mask, levels=[0.5], colors="cyan",
#             linewidths=1.0, linestyles="solid",
#             transform=ax.get_transform("pixel"))

# plt.tight_layout()
# plt.show()


tab = Table()
tab["id"] = [1, 2]  # 1 = lens, 2 = ring

print("\n=== Summary (μJy) ===")
for filt in filters:
    fcol = f"f{filt.lower()}"   # e.g. "ff115w", "ff150w"
    ecol = f"e{filt.lower()}"   # e.g. "ef115w", etc.

    tab[fcol] = [lens_fluxes[filt], ring_fluxes[filt]]
    tab[ecol] = [lens_errors[filt], ring_errors[filt]]

print(tab)
tab.write("einstein_phot.fits", overwrite=True)