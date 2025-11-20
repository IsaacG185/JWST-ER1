import numpy as np
from eazy import photoz

phot_file   = "einstein_phot_3comp_HSTJWST.fits"
translate   = "einstein.translate"
filters_res = "/opt/miniconda3/envs/new-env/lib/python3.12/site-packages/eazy/data/eazy-photoz/filters/FILTER.RES.latest"

params = {
    "CATALOG_FILE":     phot_file,
    "MAIN_OUTPUT_FILE": "einstein_eazy",
    "FILTERS_RES":      filters_res,
    "PRIOR_ABZP":       23.9,
    "Z_MIN":            0.0,
    "Z_MAX":            7.0,
    "Z_STEP":           0.01,
    "SYS_ERR":          0.03,
}

def main():
    pz = photoz.PhotoZ(
        param_file=None,
        translate_file=translate,
        params=params,
        n_proc=0,
    )

    # Actually run the fit instead of load_products
    pz.fit_catalog()

    # <<< THIS is where you can inspect things >>>
    print("coeffs shape:", pz.coeffs.shape)

    print("Available attributes on a template:")
    print([a for a in dir(pz.templates[0]) if not a.startswith("_")])

    # Example: build a continuous rest–frame model SED for object id=1
    # (just to see that everything behaves)
    idx = np.where(pz.cat["id"] == 1)[0][0]
    coeffs = pz.coeffs[idx]      # shape (NTEMPL,)
    lam    = pz.templates[0].wave   # wavelength grid (usually Angstrom)
    f_rest = np.zeros_like(lam)

    for k, tmpl in enumerate(pz.templates):
        # Many EAZY templates have something like .flux or .fnu
        f_t = getattr(tmpl, "flux", getattr(tmpl, "fnu"))
        f_rest += coeffs[k] * f_t

    print("lam grid length:", len(lam))
    print("model SED min/max f_nu:", f_rest.min(), f_rest.max())

if __name__ == "__main__":
    main()
