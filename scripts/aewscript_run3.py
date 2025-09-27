import papermill as pm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import multiprocessing

# Set GPU visibility (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

notebook_path = "output_notebook_900_datatranformations.ipynb"

# Variables and pressure level mapping
base_vars = ["cape", "crr", "d", "ie", "ishf", "lsrr", "pv", "q", "r", "sp", "tcw", "tcwv", "t", "ttr", "u", "v", "vo", "w"]
pressure_map = {
    "cape": [False],
    "crr": [False],
    "d": [900],
    "ie": [False],
    "ishf": [False],
    "lsrr": [False],
    "pv": [900],
    "q": [900],
    "r": [900],
    "sp": [False],
    "tcw": [False],
    "tcwv": [False],
    "t": [900],
    "ttr": [False],
    "u": [900],
    "v": [900],
    "vo": [900],
    "w": [900],
}

# Pressure level combinations to test
pressure_sets = [
    [900],]

# Generate tasks
tasks = []
for pset in pressure_sets:
    var_list = []
    plevel_list = []
    for var in base_vars:
        for plevel in pressure_map[var]:
            if plevel in pset or plevel is False:
                var_list.append(var)
                plevel_list.append(plevel)

    pset_str = "_".join(str(p) for p in pset)  # for naming
    output_path = f"output_notebook_transformations_{pset_str}.ipynb"
    tuner_project_name = f"tuner_runmodel_{pset_str}"
    model_save_name = f"best_modeltranformations_{pset_str}.keras"

    params = {
        "var_list": var_list,
        "plevel_list": plevel_list,
        "aew_subset": "12hr_before",
        "model_save_name": model_save_name,
        "tuner_project_name": tuner_project_name,
    }

    tasks.append((params, output_path))

# Control concurrency — run 2 at a time
max_concurrent_runs = 3  # adjust based on memory availability
with ProcessPoolExecutor(max_workers=max_concurrent_runs) as executor:
    futures = [executor.submit(pm.execute_notebook, notebook_path, output_path, parameters=params, log_output=False)
               for params, output_path in tasks]

    for future in as_completed(futures):
        try:
            result = future.result()
            print(f"✔ Completed: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")

