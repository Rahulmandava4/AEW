import papermill as pm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import multiprocessing

# Set GPU visibility (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

notebook_path = "AEW_12hr_before(300)-Copy12.ipynb"

# Variables and pressure level mapping
base_vars = ["cape", "crr", "d", "ie", "ishf", "lsrr", "pv", "q", "r", "sp", "tcw", "tcwv", "t", "ttr", "u", "v", "vo", "w"]
pressure_map = {
    "cape": [False],
    "crr": [False],
    "d": [300,850],
    "ie": [False],
    "ishf": [False],
    "lsrr": [False],
    "pv": [300,850],
    "q": [300, 850],
    "r": [300, 850],
    "sp": [False],
    "tcw": [False],
    "tcwv": [False],
    "t": [300, 850],
    "ttr": [False],
    "u": [300, 850],
    "v": [300, 850],
    "vo": [300,850],
    "w": [300, 850],
}

# Pressure level combinations to test
pressure_sets = [
    [300],
    [850],
    [300, 850],
    [300, 850],
]

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
    output_path = f"output_notebook24_{pset_str}2.ipynb"
    tuner_project_name = f"tuner_run24_{pset_str}2"
    model_save_name = f"best_model24_{pset_str}2.keras"

    params = {
        "var_list": var_list,
        "plevel_list": plevel_list,
        "aew_subset": "24hr_before",
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

