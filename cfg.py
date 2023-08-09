import os

project_name = 'DeepfakeDetection'

base_path = f"{os.getenv('HOME')}/workspace/{project_name}"
# data_path = os.path.join(base_path, "data")
results_path = os.path.join(base_path, "results")
ckpt_path = os.path.join(base_path, "checkpoints")
log_path  = os.path.join(base_path, "logs")
dataset_path = "/home/DATA/ITWM/lorenzp"
huggingface_cache_path = os.path.join(dataset_path, 'huggingface')
celeba256_path = os.path.join (dataset_path, "CelebAHQ/Img/hq/data256x256")

cfg_path = "configs"
gen      = "gen"
extract  = "extract"
detect   = "detect"

# config
cfg_gen_path = os.path.join(cfg_path, gen)
cfg_extract_path = os.path.join(cfg_path, extract)
cfg_detect_path  = os.path.join(cfg_path, detect)

# workspace
ws_gen_path = os.path.join(results_path, gen)
ws_extract_path = os.path.join(results_path, extract)
ws_detect_path  = os.path.join(results_path, detect)

# import pdb; pdb.set_trace()