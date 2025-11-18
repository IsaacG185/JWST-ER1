import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from eazy import photoz

def main():
    # --- paths / settings ---
    root = "einstein_eazy"  # MAIN_OUTPUT_FILE you used when fitting
    filters_res_file =  "/opt/miniconda3/envs/new-env/lib/python3.12/site-packages/eazy/data/eazy-photoz/filters/FILTER.RES.latest"

    params = {
        "CATALOG_FILE":     "einstein_phot.fits",
        "MAIN_OUTPUT_FILE": root,
        "FILTERS_RES":      filters_res_file,
        "PRIOR_ABZP":       23.9,   # μJy -> AB
        "Z_MIN":            0.0,
        "Z_MAX":            10.0,
        "Z_STEP":           0.01,
    }

    # Recreate PhotoZ object
    pz = photoz.PhotoZ(
        param_file=None,
        translate_file="einstein.translate",
        params=params,
        n_proc=0,      # avoid multiprocessing weirdness
    )

    # Refit catalog so all grids are consistent
    pz.fit_catalog()

    # Get standard output table in memory (don’t rely on old files)
    zout, hdu = pz.standard_output(
        save_fits=False,    # you can set True if you want to overwrite files
        get_err=True,
        n_proc=0
    )

    print(zout)

    # --- Helper: plot p(z) for a given object id ---
    def plot_pz_for_id(obj_id, ax=None, label=None):
        if ax is None:
            ax = plt.gca()

        # index of this object in the internal catalog
        idx = np.where(pz.cat["id"] == obj_id)[0][0]

        # posterior p(z) from lnp; normalize
        pz_i = np.exp(pz.lnp[idx, :])
        pz_i /= np.trapz(pz_i, pz.zgrid)

        if label is None:
            label = f"id={obj_id}"

        ax.plot(pz.zgrid, pz_i, label=label)

        # Draw vertical line at z_phot from zout table
        z_phot = zout["z_phot"][zout["id"] == obj_id][0]
        ax.axvline(z_phot, color="k", ls="--")

        return ax

    # --- 1) Plot p(z) for lens (id=1) and ring (id=2) ---
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_pz_for_id(1, ax=ax, label="Lens (id=1)")
    plot_pz_for_id(2, ax=ax, label="Ring (id=2)")
    ax.set_xlabel("z")
    ax.set_ylabel("p(z)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- 2) Plot SEDs (observed fluxes + best-fit model) for both objects ---

    # Filter pivot wavelengths (in Å)
    pivots = np.array([f.pivot for f in pz.filters])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, (obj_id, name) in zip(axes, [(1, "Lens"), (2, "Ring")]):
        idx = np.where(pz.cat["id"] == obj_id)[0][0]

        # observed fluxes and errors in μJy
        fnu = pz.cat["fnu"][idx, :]    # NFILT-length array
        efnu = pz.cat["efnu"][idx, :]
        fmodel = pz.fmodel[idx, :]     # best-fit model fluxes

        ax.errorbar(pivots, fnu, yerr=efnu, fmt="o", label="data")
        ax.plot(pivots, fmodel, "-", label="model")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("λ_obs [Å]")
        ax.set_title(f"{name} (id={obj_id})")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("fν [μJy]")
    for ax in axes:
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
