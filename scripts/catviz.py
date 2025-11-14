import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.visualization import simple_norm

# -------------------------------------------------
# USER INPUTS: edit these for your dataset
# -------------------------------------------------

filter = "f115w"  # filter name, e.g., "f115w"

# filter_list = ["f115w", "f150w", "f277w", "f444w"]

dir = "~/Documents/JWST_Project/MAST_2025-11-13T1653/JWST/jw01727-o140_t104_nircam_clear-" + filter + "/"  # directory containing your files

i2d_file  = dir + "jw01727-o140_t104_nircam_clear-" + filter + "_i2d.fits"     # your _i2d file
segm_file = dir + "jw01727-o140_t104_nircam_clear-" + filter + "_segm.fits"    # matching _segm file
cat_file  = dir + "jw01727-o140_t104_nircam_clear-" + filter + "_cat.ecsv"     # matching _cat file
# Approximate sky position of the Einstein ring (if you know it)
# If you don't know RA/Dec yet, you can skip this block
have_ring_coords = True
ring_ra  = 150.1004167   # degrees
ring_dec = 1.8930278   # degrees

# -------------------------------------------------
# 1. Read image, WCS, segmentation, and catalog
# -------------------------------------------------

# --- load i2d image + WCS ---
with fits.open(i2d_file) as hdul:
    # JWST i2d usually has SCI in extension 'SCI' or ext 1
    if "SCI" in hdul:
        sci_hdu = hdul["SCI"]
    else:
        sci_hdu = hdul[1]

    sci_data = sci_hdu.data
    wcs = WCS(sci_hdu.header)

# --- load segmentation map ---
with fits.open(segm_file) as hdul_seg:
    # segmentation is typically in the primary or first extension
    segm_data = hdul_seg[0].data if hdul_seg[0].data is not None else hdul_seg[1].data

# --- load catalog ---
cat = Table.read(cat_file, format="ascii.ecsv")

# Columns you likely have from the JWST pipeline:
# 'xcentroid', 'ycentroid', 'label' (may differ slightly, check your header)
x_col = "xcentroid"
y_col = "ycentroid"
label_col = "label"

x = cat[x_col]
y = cat[y_col]
labels = cat[label_col]

# -------------------------------------------------
# 2. Plot image + segmentation contours + catalog positions
# -------------------------------------------------
fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(projection=wcs)

# Show the i2d image
norm = simple_norm(sci_data, 'linear', percent=99.5, clip=True)
img = ax.imshow(sci_data, origin='lower', norm=norm, cmap='grey')

# Overplot segmentation map as contours
# Mask out zeros so background doesn't get contoured
segm_masked = np.ma.masked_where(segm_data == 0, segm_data)

# For many labels, contouring every single one can be busy; but for identification it’s ok
# levels are half-integers so that boundaries fall between integer labels
unique_labels = np.unique(segm_data[segm_data > 0])
levels = unique_labels - 0.5

cs = ax.contour(segm_masked, levels=levels, colors='cyan', linewidths=0.5, alpha=0.7)

# Overplot catalog centroids
# (These are in pixel coordinates, so use transform=ax.get_transform('pixel'))
ax.scatter(x, y, s=10, edgecolor='yellow', facecolor='none',
           transform=ax.get_transform('pixel'), label='Catalog sources')

# Optionally annotate a subset of sources with their label IDs (to avoid clutter)
for xi, yi, lab in tqdm(zip(x, y, labels), total=len(labels)):
    ax.text(xi, yi, str(lab),
            color='yellow', fontsize=6,
            transform=ax.get_transform('pixel'),
            ha='center', va='center')

# Plot Einstein ring position if you know RA/Dec
if have_ring_coords:
    # world_to_pixel_values returns arrays, so extract scalars
    ring_x_arr, ring_y_arr = wcs.world_to_pixel_values(ring_ra, ring_dec)

    ring_x = float(ring_x_arr)
    ring_y = float(ring_y_arr)

    ax.plot(ring_x, ring_y, marker='+', color='red',
            markersize=12, mew=2,
            transform=ax.get_transform('pixel'), label='Einstein ring')

    # Convert to nearest integer pixel indices
    ring_x_int = int(round(ring_x))
    ring_y_int = int(round(ring_y))

    # Safety check
    if (0 <= ring_x_int < segm_data.shape[1]) and (0 <= ring_y_int < segm_data.shape[0]):
        ring_label = segm_data[ring_y_int, ring_x_int]
        print(f"Segmentation label at Einstein ring position: {ring_label}")
    else:
        print("Einstein ring pixel is outside the segmentation map bounds.")

fig.colorbar(img, ax=ax, orientation='vertical', label='Flux')
ax.set_xlabel('RA')
ax.set_ylabel('Dec')
ax.set_title('NIRCam i2d with Segmentation Contours and Catalog')

ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.show()
