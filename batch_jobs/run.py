import argparse
import json
import os
import subprocess
from datetime import datetime

# from zoneinfo import ZoneInfo  # py3.9+

def dataset_recreate(config, path_config):
    batch_path = config["experiment_params"]["batch_path"]
    experiment_batch_job = config["experiment_params"]["experiment_batch_job"]
    base_path = f'{config["dataset_params"]["base_path"]}/{config["dataset_params"]["dataset"]}'
    model_name = config["model_params"]["model_name"]
    n_experiments = config["experiment_params"]["n_experiments"]
    debug_test = config["experiment_params"]["debug_test"]
    jid_a = -1
    if debug_test:
        n_experiments=1

    if "audioset" in base_path.lower():
        os.makedirs(f'{base_path}/h5s', exist_ok=True)
        jid_0 = subprocess.check_output(
                ["sbatch", "--parsable", "--array", "0-40", 
                "audioset_setup.sh", base_path], cwd=base_path
            ).strip()
        
        jid_0 = str(int(jid_0))

        jid_1 = subprocess.check_output(
                ["sbatch", "--parsable", 
                "audioset_setup.sh", base_path], cwd=base_path
            ).strip()
        
        jid_1 = str(int(jid_1))

        jid_a = subprocess.check_output(
            ["sbatch", "--parsable", f"--dependency=afterok:{jid_0}:{jid_1}",
            "audioset_VDS.sh", base_path], cwd=base_path
        ).strip()

        jid_a = str(int(jid_a))

    elif "fsd50k" in base_path.lower():
        os.makedirs(f'{base_path}/h5s', exist_ok=True)
        jid_a = subprocess.check_output(
                ["sbatch", "--parsable",
                "fsd50k_setup.sh", base_path], cwd=base_path
            ).strip()
        jid_a = str(int(jid_a))


    jid_b = subprocess.check_output(
        ["sbatch", "--parsable", "--array", f"0-{n_experiments-1}", f"--dependency=afterok:{jid_a}",
        experiment_batch_job, path_config, model_name], cwd=batch_path
    ).strip()
    return jid_a

def main(config, path_config):

    batch_path = config["experiment_params"]["batch_path"]
    experiment_batch_job = config["experiment_params"]["experiment_batch_job"]
    base_path = f'{config["dataset_params"]["base_path"]}/{config["dataset_params"]["dataset"]}'
    dest_path = base_path
    dataset = config["dataset_params"]["dataset"]
    prev_mel_params = None
    if os.path.exists(f"{base_path}/mel_params.json"):
        with open(f"{base_path}/mel_params.json", "r") as c:
            try:
                prev_mel_params = json.load(c)
            except:
                prev_mel_params = None

    n_experiments = config["experiment_params"]["n_experiments"]
    debug_test = config["experiment_params"]["debug_test"]
    if debug_test:
        n_experiments=1
    # dt = datetime.now(ZoneInfo("Europe/Helsinki"))
    dt = datetime.now()

    config["model_params"]["model_name"] += dt.strftime("_%Y-%m-%d_%H:%M")
    jid_a = None

    if "fsd50k" in dataset:
        if prev_mel_params != config["dataset_params"]["mel_params"][dataset]:
            with open(f"{dest_path}/mel_params.json", "w") as f:
                json.dump(config["dataset_params"]["mel_params"][dataset], f)
            jid_a = dataset_recreate(config=config, path_config=path_config)
    elif "audioset" in dataset:
        if prev_mel_params != config["dataset_params"]["mel_params"][dataset]:
            with open(f"{dest_path}/mel_params.json", "w") as f:
                json.dump(config["dataset_params"]["mel_params"][dataset], f)
            jid_a = dataset_recreate(config=config, path_config=path_config)
            
    if jid_a == None:
        subprocess.run(
            ["sbatch", "--parsable", "--array", f"0-{n_experiments-1}",
            experiment_batch_job, path_config, config["model_params"]["model_name"]], cwd=batch_path
        )
    return

if __name__ == '__main__':
    path_config = "/pfs/lustrep2/projappl/project_462000765/casciott/continual_learning/batch_jobs/cil_PANN14_inputs.json"
    with open(path_config, 'r') as f:
        config = json.load(f)
    main(config, path_config)
    