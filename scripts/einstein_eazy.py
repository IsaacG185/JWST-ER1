from eazy import photoz

def main():
    filters_res_file = (
        "/opt/miniconda3/envs/new-env/lib/python3.12/site-packages/"
        "eazy/data/eazy-photoz/filters/FILTER.RES.latest"  # make sure this exists
    )

    params = {
        "CATALOG_FILE":     "einstein_phot_3comp_HSTJWST.fits",
        "MAIN_OUTPUT_FILE": "einstein_eazy_3comp",
        "FILTERS_RES":      filters_res_file,
        "PRIOR_ABZP":       23.9,
        "Z_MIN":            0.0,
        "Z_MAX":            7.0,
        "Z_STEP":           0.01,
        "SYS_ERR":          0.03,
        "APPLY_PRIOR":      "n",
    }

    pz = photoz.PhotoZ(
        param_file=None,
        translate_file="einstein.translate",
        params=params,
        n_proc=0,   # single core to avoid multiprocessing issues
    )

    pz.fit_catalog()

    zout, hdu = pz.standard_output(
        save_fits=True,    # writes einstein_eazy_3comp.zout.fits etc.
        get_err=True,
        n_proc=0
    )

    print("\n=== ZOUT ===")
    print(zout)

    # Expect ids: 1 = lens, 2 = blue ring, 3 = red knots
    lens_row = zout[zout["id"] == 1][0]
    blue_row = zout[zout["id"] == 2][0]
    red_row  = zout[zout["id"] == 3][0]

    z_lens = lens_row["z_phot"]
    z_blue = blue_row["z_phot"]
    z_red  = red_row["z_phot"]

    print(f"\nLens (id=1)      z_phot = {z_lens:.3f}")
    print(f"Blue ring (id=2) z_phot = {z_blue:.3f}")
    print(f"Red knots (id=3) z_phot = {z_red:.3f}")
    print(f"Δz_blue-lens = {z_blue - z_lens:.3f}")
    print(f"Δz_red-lens  = {z_red - z_lens:.3f}")


if __name__ == "__main__":
    main()
