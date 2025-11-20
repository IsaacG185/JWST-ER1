import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from eazy import photoz

def main():
    # -----------------------------
    # USER INPUTS
    # -----------------------------
    catalog_file   = "einstein_phot_3comp_f814w.fits"  # your latest catalog
    translate_file = "einstein.translate"             # your translate file
    filters_res    = "/opt/miniconda3/envs/new-env/lib/python3.12/site-packages/eazy/data/eazy-photoz/filters/FILTER.RES.latest"    # <<< UPDATE THIS
    root           = "einstein_eazy"                  # MAIN_OUTPUT_FILE

    params = {
        "CATALOG_FILE":     catalog_file,
        "MAIN_OUTPUT_FILE": root,
        "FILTERS_RES":      filters_res,
        "PRIOR_ABZP":       23.9,
        "Z_MIN":            0.0,
        "Z_MAX":            8.0,
        "Z_STEP":           0.01,
    }

    # Initialize PhotoZ
    pz = photoz.PhotoZ(
        param_file=None,
        translate_file=translate_file,
        params=params,
        n_proc=0,      # no multiprocessing to avoid issues
    )

    # Fit catalog (re-do; NOBJ is tiny so it's quick)
    pz.fit_catalog()

    # Get standard output in memory
    zout, hdu = pz.standard_output(save_fits=False, get_err=True, n_proc=0)

    print("\n=== ZOUT TABLE ===")
    print(zout)

    # Pivot wavelengths for each filter (in Å)
    pivots = np.array([f.pivot for f in pz.filters])

    # -----------------------------
    # 1) Plot p(z) for each object
    # -----------------------------
    fig, ax = plt.subplots(figsize=(7, 4))

    for i, row in enumerate(zout):
        obj_id = row["id"]
        # EAZY stores p(z) as exp(lnp) normalized
        pz_i = np.exp(pz.lnp[i, :])
        # normalize
        pz_i /= np.trapz(pz_i, pz.zgrid)

        ax.plot(pz.zgrid, pz_i, label=f"id={obj_id}")

        # Mark z_phot
        z_phot = row["z_phot"]
        ax.axvline(z_phot, ls="--", alpha=0.6)

    ax.set_xlabel("z")
    ax.set_ylabel("p(z)")
    ax.legend()
    ax.set_title("Photometric redshift PDFs")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 2) Plot SEDs (observed vs model)
    # -----------------------------
    nobj = len(zout)
    fig, axes = plt.subplots(1, nobj, figsize=(5*nobj, 4), sharey=True)

    if nobj == 1:
        axes = [axes]

    for i, row in enumerate(zout):
        ax = axes[i]
        obj_id = row["id"]

        # Observed fluxes and errors in μJy
        fnu  = pz.cat["fnu"][i, :]    # shape: (NFILT,)
        efnu = pz.cat["efnu"][i, :]
        fmod = pz.fmodel[i, :]

        # Only plot filters actually used (nusefilt>0)
        used = fnu > 0

        ax.errorbar(pivots[used], fnu[used], yerr=efnu[used],
                    fmt="o", label="data")
        ax.plot(pivots[used], fmod[used], "-", label="model")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("λ_obs [Å]")
        ax.set_title(f"Object id={obj_id}")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("fν [μJy]")
    for ax in axes:
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
