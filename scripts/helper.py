from eazy import filters

# Use the *exact same path* you pass as FILTERS_RES in params
filters_res_file = "/opt/miniconda3/envs/new-env/lib/python3.12/site-packages/eazy/data/eazy-photoz/filters/FILTER.RES.latest"  # update this path

RES = filters.FilterFile(filters_res_file)

print("Number of filters in this file:", len(RES.filters))

# Look up the JWST NIRCam filters by name
for i, f in enumerate(RES.filters, start=1):
    name = (f.name or "").lower()
    if "jwst_nircam_f115w" in name:
        print(i, f.name)
    if "jwst_nircam_f150w" in name:
        print(i, f.name)
    if "jwst_nircam_f277w" in name:
        print(i, f.name)
    if "jwst_nircam_f444w" in name:
        print(i, f.name)
