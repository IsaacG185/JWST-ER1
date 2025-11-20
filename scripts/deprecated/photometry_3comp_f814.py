import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy.table import Table
from astropy.stats import sigma_clipped_stats

from scipy.ndimage import binary_dilation
from reproject import reproject_interp

from skimage.measure import label, regionprops  # NEW

# -----------------------------
# USER INPUTS
# -----------------------------
center_ra  = 150.100480
center_dec =   1.893035

region_width_deg = 0.001
half_width_deg   = region_width_deg / 2.0

r_lens_arcsec       = 0.30
r_ring_inner_arcsec = 0.45
r_ring_outer_arcsec = 1.10

ref_filter = "F277W"

hst_file = "~/Documents/JWST_Project/MAST_2025-11-18T2001/HST/j8pu4p010/j8pu4p010_drc.fits"  # <<< update

def dir(filter):
    return ("~/Documents/JWST_Project/MAST_2025-11-13T1653/JWST/"
            "jw01727-o140_t104_nircam_clear-" + filter + "/")

filter_files = {
    "F115W": dir("f115w") + "jw01727-o140_t104_nircam_clear-f115w_i2d.fits",
    "F150W": dir("f150w") + "jw01727-o140_t104_nircam_clear-f150w_i2d.fits",
    "F277W": dir("f277w") + "jw01727-o140_t104_nircam_clear-f277w_i2d.fits",
    "F444W": dir("f444w") + "jw01727-o140_t104_nircam_clear-f444w_i2d.fits",
}

# -----------------------------
# HELPERS
# -----------------------------
def get_pixscale_deg(header):
    if "CDELT1" in header:
        return abs(header["CDELT1"])
    cd11 = header.get("CD1_1", None)
    cd12 = header.get("CD1_2", None)
    if cd11 is not None and cd12 is not None:
        return np.sqrt(cd11**2 + cd12**2)
    pc11   = header.get("PC1_1", None)
    pc12   = header.get("PC1_2", None)
    cdelt1 = header.get("CDELT1", None)
    if pc11 is not None and pc12 is not None and cdelt1 is not None:
        return abs(cdelt1) * np.sqrt(pc11**2 + pc12**2)
    print("Warning: using default 0.031\"/pix")
    return 0.031 / 3600.0

def aperture_flux(sub_sci, sub_err, mask, pixar_sr):
    sci_mjysr = sub_sci[mask]
    err_mjysr = sub_err[mask]
    flux_uJy_pix = sci_mjysr * (pixar_sr * 1e6)
    err_uJy_pix  = err_mjysr  * (pixar_sr * 1e6)
    flux_tot = flux_uJy_pix.sum()
    err_tot  = np.sqrt((err_uJy_pix**2).sum())
    return flux_tot, err_tot

def aperture_flux_uJy(data_uJy, err_uJy, mask):
    vals = data_uJy[mask]
    errs = err_uJy[mask]
    flux_tot = vals.sum()
    err_tot  = np.sqrt((errs**2).sum())
    return flux_tot, err_tot

# -----------------------------
# 1. MASKS FROM F277W
# -----------------------------
ref_fname = filter_files[ref_filter]
print(f"Using {ref_filter} as reference image for masks: {ref_fname}")

with fits.open(ref_fname) as hdul:
    sci_hdu = hdul["SCI"] if "SCI" in hdul else hdul[1]
    sci_data_ref = sci_hdu.data
    header_ref   = sci_hdu.header
    wcs_ref      = WCS(header_ref)
    err_data_ref = hdul["ERR"].data
    pixar_sr_ref = header_ref.get("PIXAR_SR", None)
    if pixar_sr_ref is None:
        raise RuntimeError(f"PIXAR_SR not found in {ref_fname}")

ny, nx = sci_data_ref.shape

pixscale_deg    = get_pixscale_deg(header_ref)
pixscale_arcsec = pixscale_deg * 3600.0

x_arr, y_arr = wcs_ref.world_to_pixel_values(center_ra, center_dec)
x_center = float(x_arr)
y_center = float(y_arr)

half_width_pix = half_width_deg / pixscale_deg
x_min = int(np.round(x_center - half_width_pix))
x_max = int(np.round(x_center + half_width_pix))
y_min = int(np.round(y_center - half_width_pix))
y_max = int(np.round(y_center + half_width_pix))

x_min = max(x_min, 0); y_min = max(y_min, 0)
x_max = min(x_max, nx); y_max = min(y_max, ny)

print(f"Cutout bounds (x: {x_min}-{x_max}, y: {y_min}-{y_max})")

sub_ref     = sci_data_ref[y_min:y_max, x_min:x_max]
sub_err_ref = err_data_ref[y_min:y_max, x_min:x_max]

ny_sub, nx_sub = sub_ref.shape
yy, xx = np.indices((ny_sub, nx_sub))

x0 = x_center - x_min
y0 = y_center - y_min

r_pix = np.sqrt((xx - x0)**2 + (yy - y0)**2)

r_lens_pix       = r_lens_arcsec       / pixscale_arcsec
r_ring_inner_pix = r_ring_inner_arcsec / pixscale_arcsec
r_ring_outer_pix = r_ring_outer_arcsec / pixscale_arcsec

ring_shell = (r_pix >= r_ring_inner_pix) & (r_pix <= r_ring_outer_pix)

mean, med, std = sigma_clipped_stats(sub_ref[ring_shell], sigma=2.5)
thr = med + 1.5*std

ring_initial = ring_shell & (sub_ref > thr)
ring_mask = binary_dilation(ring_initial, iterations=1)

lens_base_mask = (r_pix <= r_lens_pix)
lens_mask = lens_base_mask & (~ring_mask)

# -----------------------------
# 1b. SPLIT RING INTO BLUE / RED COMPONENTS
# -----------------------------
labels = label(ring_mask)
regions = regionprops(labels)

# Heuristic: large-area regions → blue ring, small-area → red knots
areas = np.array([r.area for r in regions])
# choose a threshold; you can tweak this
area_thresh = np.percentile(areas, 60)  # top 40% = blue, bottom 60% = red

blue_ids = [1, 3]   # e.g. [1, 3]
red_ids  = [2, 4]   # e.g. [2, 4]

# Initialize masks
ring_blue_mask = np.zeros_like(ring_mask, dtype=bool)
ring_red_mask  = np.zeros_like(ring_mask, dtype=bool)

for i, r in enumerate(regions, start=1):
    if i in blue_ids:
        ring_blue_mask[labels == r.label] = True
    elif i in red_ids:
        ring_red_mask[labels == r.label] = True

# Sanity check plot
fig, ax = plt.subplots(figsize=(5, 5))
norm = simple_norm(sub_ref, "sqrt", percent=99.5, clip=True)
ax.imshow(sub_ref, origin="lower", norm=norm, cmap="gray")
ax.contour(lens_mask,      levels=[0.5], colors="yellow",  linewidths=1.0)
ax.contour(ring_blue_mask, levels=[0.5], colors="cyan",    linewidths=1.0)
ax.contour(ring_red_mask,  levels=[0.5], colors="magenta", linewidths=1.0)
ax.plot(x0, y0, "r+", ms=10, mew=2)
ax.set_title(f"{ref_filter} lens / blue ring / red knots (manual)")
ax.set_xlabel("x (pix)")
ax.set_ylabel("y (pix)")
plt.tight_layout()
plt.show()

# -----------------------------
# 2. JWST FLUXES FOR 3 OBJECTS
# -----------------------------
filters = ["F115W", "F150W", "F277W", "F444W"]

lens_fluxes = {}
lens_errors = {}
blue_fluxes = {}
blue_errors = {}
red_fluxes  = {}
red_errors  = {}

for filt in filters:
    fname = filter_files[filt]
    print(f"\n=== {filt} ===")
    print(f"Loading {fname}")

    with fits.open(fname) as hdul:
        sci_hdu = hdul["SCI"] if "SCI" in hdul else hdul[1]
        sci_data = sci_hdu.data
        header   = sci_hdu.header
        err_data = hdul["ERR"].data
        pixar_sr = header.get("PIXAR_SR", None)
        if pixar_sr is None:
            raise RuntimeError(f"PIXAR_SR not found in {fname}")

    sub_sci = sci_data[y_min:y_max, x_min:x_max]
    sub_err = err_data[y_min:y_max, x_min:x_max]

    if sub_sci.shape != ring_mask.shape:
        raise RuntimeError(f"Shape mismatch in {filt}: {sub_sci.shape} vs {ring_mask.shape}")

    lens_flux, lens_err = aperture_flux(sub_sci, sub_err, lens_mask,       pixar_sr)
    blue_flux, blue_err = aperture_flux(sub_sci, sub_err, ring_blue_mask,  pixar_sr)
    red_flux,  red_err  = aperture_flux(sub_sci, sub_err, ring_red_mask,   pixar_sr)

    lens_fluxes[filt] = lens_flux
    lens_errors[filt] = lens_err
    blue_fluxes[filt] = blue_flux
    blue_errors[filt] = blue_err
    red_fluxes[filt]  = red_flux
    red_errors[filt]  = red_err

    print(f"Lens       (μJy): {lens_flux:.3g} ± {lens_err:.3g}")
    print(f"Blue ring  (μJy): {blue_flux:.3g} ± {blue_err:.3g}")
    print(f"Red knots  (μJy): {red_flux:.3g} ± {red_err:.3g}")

# -----------------------------
# 3. HST F814W (same split masks)
# -----------------------------
print(f"\n=== F814W (HST) ===")
print(f"Loading {hst_file}")

with fits.open(hst_file) as hdul:
    sci_hdu = hdul[1]
    hst_data = sci_hdu.data.astype(float)
    hst_hdr  = sci_hdu.header
    hst_wcs  = WCS(hst_hdr)

    if "ERR" in hdul:
        err_data = hdul["ERR"].data.astype(float)
    elif "WHT" in hdul:
        wht = hdul["WHT"].data.astype(float)
        err_data = np.zeros_like(wht)
        good = wht > 0
        err_data[good] = 1.0 / np.sqrt(wht[good])
    else:
        raise RuntimeError("No ERR or WHT extension in HST file.")

    photflam = hst_hdr["PHOTFLAM"]
    photplam = hst_hdr["PHOTPLAM"]

jwst_hdr = header_ref.copy()
jwst_hdr["NAXIS1"] = nx_sub
jwst_hdr["NAXIS2"] = ny_sub

hst_reproj, _     = reproject_interp((hst_data, hst_wcs), jwst_hdr,
                                     shape_out=(ny_sub, nx_sub))
hst_err_reproj, _ = reproject_interp((err_data, hst_wcs), jwst_hdr,
                                     shape_out=(ny_sub, nx_sub))

c_angs_per_s = 2.99792458e18
conv = photflam * (photplam**2) / (c_angs_per_s * 1e-23) * 1e6

hst_uJy     = hst_reproj * conv
hst_err_uJy = hst_err_reproj * conv

lens_f814w, lens_e814w = aperture_flux_uJy(hst_uJy, hst_err_uJy, lens_mask)
blue_f814w, blue_e814w = aperture_flux_uJy(hst_uJy, hst_err_uJy, ring_blue_mask)
red_f814w,  red_e814w  = aperture_flux_uJy(hst_uJy, hst_err_uJy, ring_red_mask)

print(f"Lens F814W      (μJy): {lens_f814w:.3g} ± {lens_e814w:.3g}")
print(f"Blue ring F814W (μJy): {blue_f814w:.3g} ± {blue_e814w:.3g}")
print(f"Red knots F814W (μJy): {red_f814w:.3g} ± {red_e814w:.3g}")

# -----------------------------
# 4. OUTPUT TABLE FOR EAZY (3 COMPONENTS)
# -----------------------------
tab = Table()
tab["id"] = [1, 2, 3]  # 1=lens, 2=blue ring, 3=red knots

for filt in filters:
    fcol = f"f{filt.lower()}"
    ecol = f"e{filt.lower()}"
    tab[fcol] = [lens_fluxes[filt], blue_fluxes[filt], red_fluxes[filt]]
    tab[ecol] = [lens_errors[filt], blue_errors[filt], red_errors[filt]]

tab["ff814w"] = [lens_f814w, blue_f814w, red_f814w]
tab["ef814w"] = [lens_e814w, blue_e814w, red_e814w]

print("\nFinal photometry table (3 components, including F814W):")
print(tab)

tab.write("einstein_phot_3comp_f814w.fits", overwrite=True)
print("\nWrote photometry to einstein_phot_3comp_f814w.fits")
