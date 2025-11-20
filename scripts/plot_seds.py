import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from eazy.filters import FilterFile

# --- paths ---
filters_res_file = "/opt/miniconda3/envs/new-env/lib/python3.12/site-packages/eazy/data/eazy-photoz/filters/FILTER.RES.latest"
phot_file        = "einstein_phot_3comp_HSTJWST.fits"

# SAME mapping as in einstein.translate  ⬇⬇  (fill these in as you used them)
col_to_fid = {
    "ff115w": 364,
    "ff150w": 365,
    "ff277w": 375,
    "ff444w": 377,
    "ff435w":  79,   # <-- your F435W ID
    "ff606w":  95,   # <-- your F606W ID
    "ff814w":   6,   # <-- your ACS F814W ID
    "ff160w": 217,   # <-- your F160W ID
}

# ------------------------------------------------------------------
# Helper: effective wavelength from filter curve
# ------------------------------------------------------------------
def effective_lambda(filt):
    """
    Compute λ_eff = Σ(λ T) / Σ(T) using the FilterDefinition
    """
    # wavelength array is a Quantity in Angstrom
    lam = np.array(filt.wave)
    if hasattr(lam, "value"):   # strip astropy units
        lam = lam.value

    # throughput can be 'throughput' or 'thru' depending on version
    if hasattr(filt, "throughput"):
        T = np.array(filt.throughput)
    elif hasattr(filt, "thru"):
        T = np.array(filt.thru)
    else:
        raise AttributeError("Filter has no 'throughput' or 'thru' attribute")

    return np.sum(lam * T) / np.sum(T)   # Angstrom

# ------------------------------------------------------------------
# Load filters and photometry
# ------------------------------------------------------------------
F    = FilterFile(filters_res_file)
phot = Table.read(phot_file)

# Bands/columns in the same order you want them plotted
bands = ["ff435w", "ff606w", "ff814w", "ff160w",
         "ff115w", "ff150w", "ff277w", "ff444w"]

# Effective wavelengths for each band (Å)
l_eff = np.array([effective_lambda(F[col_to_fid[c]]) for c in bands])

objs = {
    1: {"label": "Lens",      "color": "tab:blue"},
    2: {"label": "Blue ring", "color": "tab:orange"},
    3: {"label": "Red knots", "color": "tab:green"},
}

plt.figure(figsize=(8, 5))

for obj_id, info in objs.items():
    row = phot[phot["id"] == obj_id][0]

    f = np.array([row[c] for c in bands])             # μJy
    e = np.array([row["e" + c[1:]] for c in bands])   # μJy

    # Convert μJy → AB mag, guard against nonpositive flux
    good = f > 0
    m  = np.full_like(f,  np.nan, dtype=float)
    dm = np.full_like(e,  np.nan, dtype=float)

    m[good]  = -2.5 * np.log10(f[good]) + 23.9
    dm[good] = (2.5 / np.log(10)) * (e[good] / f[good])

    plt.errorbar(l_eff[good]/1e4, m[good], yerr=dm[good],
                 fmt="o-", color=info["color"],
                 label=f"{info['label']} (id={obj_id})")

plt.gca().invert_yaxis()
plt.xlabel("Wavelength [µm]")
plt.ylabel("AB magnitude")
# plt.gca().invert_yaxis()
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/seds_plot.png", dpi=150)
plt.show()
