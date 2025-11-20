import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy.stats import sigma_clipped_stats

from scipy.ndimage import binary_dilation
from reproject import reproject_interp
from skimage.measure import label, regionprops

from photutils.centroids import centroid_com
from scipy.ndimage import shift

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

home_dir = "~/Documents/JWST_Project/"

# JWST i2d files (same paths as your photometry)
def jw_dir(filt):
    return (home_dir + "MAST_2025-11-13T1653/JWST/jw01727-o140_t104_nircam_clear-" + filt + "/")

jwst_files = {
    "F115W": jw_dir("f115w") + "jw01727-o140_t104_nircam_clear-f115w_i2d.fits",
    "F150W": jw_dir("f150w") + "jw01727-o140_t104_nircam_clear-f150w_i2d.fits",
    "F277W": jw_dir("f277w") + "jw01727-o140_t104_nircam_clear-f277w_i2d.fits",
    "F444W": jw_dir("f444w") + "jw01727-o140_t104_nircam_clear-f444w_i2d.fits",
}

# HST drizzled science images (drc/drz) – UPDATE THESE PATHS
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

sub_ref = sci_data_ref[y_min:y_max, x_min:x_max]
ny_sub, nx_sub = sub_ref.shape

# WCS for the cutout (this is the reference WCS)
wcs_cut = wcs_ref.slice((slice(y_min, y_max), slice(x_min, x_max)))
jwst_hdr = wcs_cut.to_header()
jwst_hdr["NAXIS1"] = nx_sub
jwst_hdr["NAXIS2"] = ny_sub

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

# Label ring components
labels = label(ring_mask)
regions = regionprops(labels)

print("\nConnected components in ring_mask:")
for i, r in enumerate(regions, start=1):
    y_c, x_c = r.centroid
    print(f"  ID {i}: area={r.area}, centroid=({x_c:.1f}, {y_c:.1f})")

# >>> IMPORTANT: use the SAME IDs here as in your photometry script
blue_ids = [1, 3, 4, 5]   
red_ids  = [2, 6] 

ring_blue_mask = np.zeros_like(ring_mask, dtype=bool)
ring_red_mask  = np.zeros_like(ring_mask, dtype=bool)

for i, r in enumerate(regions, start=1):
    if i in blue_ids:
        ring_blue_mask[labels == r.label] = True
    elif i in red_ids:
        ring_red_mask[labels == r.label] = True

# -----------------------------
# 2. GATHER IMAGES ONTO THE SAME GRID
# -----------------------------
images = {}
order = ["F435W", "F606W", "F814W", "F160W",
         "F115W", "F150W", "F277W", "F444W"]


# Reference band: F277W
images["F277W"] = sub_ref

# JWST: just slice to the same cutout bounds
for filt in ["F115W", "F150W", "F444W"]:
    fname = jwst_files[filt]
    with fits.open(fname) as hdul:
        sci_hdu = hdul["SCI"] if "SCI" in hdul else hdul[1]
        sci_data = sci_hdu.data
        wcs_filt = WCS(sci_hdu.header)

    # Reproject this filter onto the F277W cutout WCS
    reproj, _ = reproject_interp((sci_data, wcs_filt),
                                 jwst_hdr,
                                 shape_out=(ny_sub, nx_sub))
    images[filt] = reproj

# HST: reproject each to the JWST cutout grid
for filt, hst_file in hst_files.items():
    with fits.open(hst_file) as hdul:
        sci_hdu = hdul[1]
        hst_data = sci_hdu.data.astype(float)
        hst_hdr  = sci_hdu.header
        hst_wcs  = WCS(hst_hdr)

    reproj, _ = reproject_interp((hst_data, hst_wcs),
                             jwst_hdr,
                             shape_out=(ny_sub, nx_sub))
    images[filt] = reproj

# if "F814W" in images:
#     img814 = images["F814W"]

#     # small box around expected center to avoid picking up ring/star
#     r_box = 16  # pixels
#     yy_c, xx_c = np.indices(img814.shape)
#     box = (np.abs(xx_c - x0) <= r_box) & (np.abs(yy_c - y0) <= r_box)

#     # use only that box for centroid
#     cut = np.zeros_like(img814)
#     cut[box] = img814[box]

#     try:
#         y814, x814 = centroid_com(cut)
#         dy = y0 - y814
#         dx = x0 - x814
#         print(f"Shifting F814W by (dx, dy) = ({dx:.2f}, {dy:.2f}) pixels")

#         images["F814W"] = shift(img814, shift=(dy, dx), order=1, mode="nearest")
#     except Exception as e:
#         print(f"Could not compute centroid for F814W: {e}")

if "F814W" in images:
    img814 = images["F814W"]
    dx = 8
    dy = 0
    print(f"Shifting F814W by (dx, dy) = ({dx:.2f}, {dy:.2f}) pixels")

    images["F814W"] = shift(img814, shift=(dy, dx), order=1, mode="nearest")

# -----------------------------
# 3. MAKE A 2×4 GRID WITH MASKS OVERLAID
# -----------------------------
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.ravel()

for ax, filt in zip(axes, order):
    img = images[filt]
    finite = np.isfinite(img)
    if finite.sum() == 0:
        ax.imshow(np.zeros_like(img), origin="lower", cmap="gray")
        ax.text(0.5, 0.5, "No data", color="red",
                ha="center", va="center", transform=ax.transAxes)
    else:
        norm = simple_norm(img[finite], "sqrt", percent=99.0, clip=True)
        ax.imshow(img, origin="lower", norm=norm, cmap="gray")
        ax.contour(lens_mask,      levels=[0.5], colors="yellow")
        ax.contour(ring_blue_mask, levels=[0.5], colors="cyan")
        ax.contour(ring_red_mask,  levels=[0.5], colors="magenta")


    ax.set_title(filt)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig("figures/masks_all_filters.png", dpi=150)
plt.show()
