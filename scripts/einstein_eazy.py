from eazy import photoz

def main():
    filters_res_file = "/opt/miniconda3/envs/new-env/lib/python3.12/site-packages/eazy/data/eazy-photoz/filters/FILTER.RES.latest"  # update this path


    params = {
        "CATALOG_FILE":     "einstein_phot.fits",
        "MAIN_OUTPUT_FILE": "einstein_eazy",
        "FILTERS_RES":      filters_res_file,
        "PRIOR_ABZP":       23.9,
        "Z_MIN":            0.0,
        "Z_MAX":            10.0,
        "Z_STEP":           0.01,
    }

    pz = photoz.PhotoZ(
        param_file=None,
        translate_file="einstein.translate",
        params=params,
        n_proc=0,          # <<< IMPORTANT: turn off multiprocessing
    )

    pz.fit_catalog()

    zout, hdu = pz.standard_output(
        save_fits=True,    # write einstein_eazy.zout.fits + .data.fits
        get_err=True,
        n_proc=0           # also keep this single-core
    )

    print(zout)

    # Extract lens & ring entries
    lens_row = zout[zout["id"] == 1][0]
    ring_row = zout[zout["id"] == 2][0]

    # EAZY-py calls the main best-fit redshift `z_phot` (and also stores `z_ml`)
    z_lens = lens_row["z_phot"]
    z_ring = ring_row["z_phot"]

    print(f"Lens z_phot   = {z_lens:.3f}")
    print(f"Ring z_phot   = {z_ring:.3f}")
    print(f"Δz (ring - lens) = {z_ring - z_lens:.3f}")


if __name__ == "__main__":
    main()
