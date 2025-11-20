import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy.table import Table
from astropy.stats import sigma_clipped_stats

from scipy.ndimage import binary_dilation, shift   # <<< NEW: import shift
from reproject import reproject_interp
from skimage.measure import label, regionprops

# -----------------------------
# USER INPUTS
# -----------------------------
center_ra  = 150.100480
center_dec =   1.893035

region_width_deg = 0.001
half_width_deg   = region_width_deg / 2.0

r_lens_arcsec       = 0.27
r_ring_inner_arcsec = 0.6
r_ring_outer_arcsec = 1.0

ref_filter = "F277W"
home_dir = "~/Documents/JWST_Project/"

# JWST i2d files
def jw_dir(filt):
    return (home_dir + "MAST_2025-11-13T1653/JWST/jw01727-o140_t104_nircam_clear-" + filt + "/")

jwst_files = {
    "F115W": jw_dir("f115w") + "jw01727-o140_t104_nircam_clear-f115w_i2d.fits",
    "F150W": jw_dir("f150w") + "jw01727-o140_t104_nircam_clear-f150w_i2d.fits",
    "F277W": jw_dir("f277w") + "jw01727-o140_t104_nircam_clear-f277w_i2d.fits",
    "F444W": jw_dir("f444w") + "jw01727-o140_t104_nircam_clear-f444w_i2d.fits",
}

# HST drizzled science images (drc/drz)
hst_files = {
    "F435W": home_dir + "MAST_2025-11-19T1909/HST/hst_17802_dp_acs_wfc_f435w_jffudp/hst_17802_dp_acs_wfc_f435w_jffudp_drc.fits", 
    "F606W": home_dir + "MAST_2025-11-19T1909/HST/hst_17802_dp_acs_wfc_f606w_jffudp/hst_17802_dp_acs_wfc_f606w_jffudp_drc.fits", 
    "F814W": home_dir + "MAST_2025-11-19T1909/HST/j8pu4p010/j8pu4p010_drc.fits",
    "F160W": home_dir + "MAST_2025-11-19T1909/HST/hst_14114_05_wfc3_ir_f160w_icxe05/hst_14114_05_wfc3_ir_f160w_icxe05_drz.fits"
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

    # MJy/sr * sr → MJy; MJy → µJy with 1e12
    flux_uJy_pix = sci_mjysr * pixar_sr * 1e12
    err_uJy_pix  = err_mjysr  * pixar_sr * 1e12

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
# 1. BUILD MASKS FROM JWST F277W
# -----------------------------
ref_fname = jwst_files[ref_filter]
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

# WCS/header for the cutout grid (reference for all reprojections)  <<< NEW
wcs_cut = wcs_ref.slice((slice(y_min, y_max), slice(x_min, x_max)))
jwst_hdr = wcs_cut.to_header()
jwst_hdr["NAXIS1"] = nx_sub
jwst_hdr["NAXIS2"] = ny_sub

# -----------------------------
# 1b. LABEL RING COMPONENTS, THEN MANUALLY ASSIGN BLUE/RED
# -----------------------------
labels = label(ring_mask)
regions = regionprops(labels)

print("\nConnected components in ring_mask:")
for i, r in enumerate(regions, start=1):
    y_c, x_c = r.centroid
    print(f"  ID {i}: area={r.area}, centroid=({x_c:.1f}, {y_c:.1f})")

# (Optional diagnostic plot — keep if you like)
fig, ax = plt.subplots(figsize=(5, 5))
norm = simple_norm(sub_ref, "sqrt", percent=99.0, clip=True)
ax.imshow(sub_ref, origin="lower", norm=norm, cmap="gray")
ax.contour(ring_mask, levels=[0.5], colors="white", linewidths=0.8)
for i, r in enumerate(regions, start=1):
    y_c, x_c = r.centroid
    ax.text(x_c, y_c, str(i), color="red", fontsize=9,
            ha="center", va="center")
ax.plot(x0, y0, "r+", ms=10, mew=2)
ax.set_title(f"{ref_filter} components (IDs)")
plt.tight_layout()
plt.show()

# Use the IDs you decided on visually
blue_ids = [1, 3, 4, 5]
red_ids  = [2, 6]

ring_blue_mask = np.zeros_like(ring_mask, dtype=bool)
ring_red_mask  = np.zeros_like(ring_mask, dtype=bool)

for i, r in enumerate(regions, start=1):
    if i in blue_ids:
        ring_blue_mask[labels == r.label] = True
    elif i in red_ids:
        ring_red_mask[labels == r.label] = True

# Sanity-check plot
fig, ax = plt.subplots(figsize=(5, 5))
norm = simple_norm(sub_ref, "sqrt", percent=99.0, clip=True)
ax.imshow(sub_ref, origin="lower", norm=norm, cmap="gray")
ax.contour(lens_mask,       levels=[0.5], colors="yellow",  linewidths=1.0)
ax.contour(ring_blue_mask,  levels=[0.5], colors="cyan",    linewidths=1.0)
ax.contour(ring_red_mask,   levels=[0.5], colors="magenta", linewidths=1.0)
ax.plot(x0, y0, "r+", ms=10, mew=2)
ax.set_title(f"{ref_filter} lens / blue ring / red knots (manual)")
plt.tight_layout()
plt.savefig("figures/masks_manual_assignment.png", dpi=150)
plt.show()

# -----------------------------
# 2. JWST PHOTOMETRY (3 OBJECTS) – NOW USING REPROJECTION  <<< NEW
# -----------------------------
jw_filters = ["F115W", "F150W", "F277W", "F444W"]

lens_fluxes = {}
lens_errors = {}
blue_fluxes = {}
blue_errors = {}
red_fluxes  = {}
red_errors  = {}

for filt in jw_filters:
    fname = jwst_files[filt]
    print(f"\n=== {filt} (JWST) ===")
    print(f"Loading {fname}")

    with fits.open(fname) as hdul:
        sci_hdu = hdul["SCI"] if "SCI" in hdul else hdul[1]
        sci_data = sci_hdu.data
        header   = sci_hdu.header
        err_data = hdul["ERR"].data
        pixar_sr = header.get("PIXAR_SR", None)
        if pixar_sr is None:
            raise RuntimeError(f"PIXAR_SR not found in {fname}")
        wcs_filt = WCS(header)

    if filt == ref_filter:
        # Reference band: use the already-cut-out F277W arrays
        sub_sci = sub_ref
        sub_err = sub_err_ref
    else:
        # Reproject this filter onto the F277W cutout WCS
        sub_sci, _ = reproject_interp((sci_data, wcs_filt),
                                      jwst_hdr,
                                      shape_out=(ny_sub, nx_sub))
        sub_err, _ = reproject_interp((err_data, wcs_filt),
                                      jwst_hdr,
                                      shape_out=(ny_sub, nx_sub))

    if sub_sci.shape != ring_mask.shape:
        raise RuntimeError(f"Shape mismatch in {filt}: {sub_sci.shape} vs {ring_mask.shape}")

    lens_flux, lens_err = aperture_flux(sub_sci, sub_err, lens_mask,      pixar_sr)
    blue_flux, blue_err = aperture_flux(sub_sci, sub_err, ring_blue_mask, pixar_sr)
    red_flux,  red_err  = aperture_flux(sub_sci, sub_err, ring_red_mask,  pixar_sr)

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
# 3. HST PHOTOMETRY (F435W, F606W, F814W, F160W)
# -----------------------------
hst_lens_flux  = {}
hst_lens_err   = {}
hst_blue_flux  = {}
hst_blue_err   = {}
hst_red_flux   = {}
hst_red_err    = {}

c_angs_per_s = 2.99792458e18

for filt, hst_file in hst_files.items():
    print(f"\n=== {filt} (HST) ===")
    print(f"Loading {hst_file}")

    with fits.open(hst_file) as hdul:
        sci_hdu = hdul[1]  # drizzled SCI
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

    # Reproject onto JWST cutout grid
    hst_reproj, _     = reproject_interp((hst_data, hst_wcs), jwst_hdr,
                                         shape_out=(ny_sub, nx_sub))
    hst_err_reproj, _ = reproject_interp((err_data, hst_wcs), jwst_hdr,
                                         shape_out=(ny_sub, nx_sub))

    # counts/s → μJy  (AB-like conversion)
    conv = photflam * (photplam**2) / (c_angs_per_s * 1e-23) * 1e6
    hst_uJy     = hst_reproj * conv
    hst_err_uJy = hst_err_reproj * conv

    # <<< NEW: empirical alignment fix for F814W
    if filt == "F814W":
        dx, dy = 8.0, 0.0   # shift right by 8 pixels
        print(f" Shifting F814W by (dx, dy) = ({dx}, {dy}) pixels")
        hst_uJy     = shift(hst_uJy,     shift=(dy, dx), order=1, mode="nearest")
        hst_err_uJy = shift(hst_err_uJy, shift=(dy, dx), order=1, mode="nearest")

    lf, le = aperture_flux_uJy(hst_uJy, hst_err_uJy, lens_mask)
    bf, be = aperture_flux_uJy(hst_uJy, hst_err_uJy, ring_blue_mask)
    rf, re = aperture_flux_uJy(hst_uJy, hst_err_uJy, ring_red_mask)

    hst_lens_flux[filt] = lf
    hst_lens_err[filt]  = le
    hst_blue_flux[filt] = bf
    hst_blue_err[filt]  = be
    hst_red_flux[filt]  = rf
    hst_red_err[filt]   = re

    print(f"Lens {filt}      (μJy): {lf:.3g} ± {le:.3g}")
    print(f"Blue ring {filt} (μJy): {bf:.3g} ± {be:.3g}")
    print(f"Red knots {filt} (μJy): {rf:.3g} ± {re:.3g}")

# -----------------------------
# 4. BUILD OUTPUT TABLE FOR EAZY (3 OBJECTS × 8 BANDS)
# -----------------------------
tab = Table()
tab["id"] = [1, 2, 3]  # 1=lens, 2=blue ring, 3=red knots

# JWST bands
for filt in jw_filters:
    fcol = f"ff{filt[1:].lower()}"  # e.g. F115W -> ff115w
    ecol = f"ef{filt[1:].lower()}"
    tab[fcol] = [lens_fluxes[filt], blue_fluxes[filt], red_fluxes[filt]]
    tab[ecol] = [lens_errors[filt], blue_errors[filt], red_errors[filt]]

# HST bands
for filt in ["F435W", "F606W", "F814W", "F160W"]:
    short = filt[1:].lower()   # "435w", "606w", etc.
    fcol = f"ff{short}"
    ecol = f"ef{short}"
    tab[fcol] = [hst_lens_flux[filt], hst_blue_flux[filt], hst_red_flux[filt]]
    tab[ecol] = [hst_lens_err[filt],  hst_blue_err[filt],  hst_red_err[filt]]

print("\nFinal photometry table (3 components, HST+JWST):")
print(tab)

# ---- quick S/N diagnostics ----
print("\nSignal-to-noise by band (JWST + HST):")

def snr(flux, err):
    return flux/err if err > 0 else np.nan

for i, name in enumerate(["Lens", "Blue ring", "Red knots"]):
    print(f"\n{name} (id={i+1}):")
    for col in tab.colnames:
        if col.startswith("ff"):
            ecol = "e" + col[1:]
            f = tab[col][i]
            e = tab[ecol][i]
            print(f"  {col:6s}: f = {f: .3e} μJy, e = {e: .3e}, S/N = {snr(f,e): .2f}")

# ---- De-weight contaminated blue bands for the ring components ----
# ids: 1 = lens, 2 = blue ring, 3 = red knots
for band in ["ff435w", "ff606w"]:
    eband = "e" + band[1:]
    # indices 1 and 2 correspond to id=2 and id=3 in the table
    for i in (1, 2):
        f = tab[band][i]
        e = tab[eband][i]
        # Treat as an upper limit: set flux ~0 with a large error
        tab[band][i]  = 0.0
        tab[eband][i] = max(e, 5.0*abs(f))  # 5σ upper limit


outname = "einstein_phot_3comp_HSTJWST.fits"
tab.write(outname, overwrite=True)
print(f"\nWrote photometry to {outname}")

# Combine blue+red masks if you want the full ring
full_ring_mask = ring_blue_mask | ring_red_mask   # or just `ring_mask` if you prefer

# Radii of all pixels in the ring, in pixels
r_ring_pix = r_pix[full_ring_mask]

# Robust estimate of Einstein radius in pixels
R_E_pix = np.median(r_ring_pix)

# Convert to arcsec
theta_E_arcsec = R_E_pix * pixscale_arcsec
print(f"Estimated Einstein radius: {theta_E_arcsec:.3f} arcsec")