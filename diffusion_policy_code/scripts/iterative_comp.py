import os
from tqdm import tqdm

data_dir = '/media/yixuan_2T/dynamic_repr/NeuRay/data/real_plan/scissors_eval'

# iteratively compress all subfolders in data_dir
for subfolder in tqdm(sorted(os.listdir(data_dir))):
    if os.path.isdir(os.path.join(data_dir, subfolder)):
        os.system(f'zip -r {os.path.join(data_dir, subfolder)}.zip {os.path.join(data_dir, subfolder)}')
