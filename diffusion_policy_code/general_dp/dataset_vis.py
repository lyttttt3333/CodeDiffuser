import sys
import os
from pathlib import Path
import glob
from omegaconf import OmegaConf
import hydra
from tqdm import tqdm
import h5py
import numpy as np
import open3d as o3d
import copy
import time

from diffusion_policy.common.kinematics_utils import KinHelper
from d3fields.utils.draw_utils import np2o3d

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    assert cfg.is_real == False, "This script is for simulation only"
    if hasattr(cfg.task, "dataset_dir"):
        dataset_dir = cfg.task.dataset_dir
        sys.path.append(dataset_dir)
    dataset_dir = Path(dataset_dir).expanduser()
    episodes_paths = glob.glob(os.path.join(dataset_dir, "episode_*.hdf5"))
    episodes_stem_name = [Path(path).stem for path in episodes_paths]
    episodes_idx = [int(stem_name.split("_")[-1]) for stem_name in episodes_stem_name]
    episodes_idx = sorted(episodes_idx)

    kin_helper = KinHelper(robot_name=cfg.robot_name)
    assert episodes_idx, "No episodes found in the dataset directory"

    for epi_idx in tqdm(episodes_idx, desc=f"Loading episodes"):
        dataset_path = os.path.join(dataset_dir, f"episode_{epi_idx}.hdf5")
        with h5py.File(dataset_path) as file:
            joint_action = file["joint_action"]

            for i in range(joint_action.shape[0]):
                curr_qpos = np.append(joint_action[i], joint_action[i, -1])
                pcd = kin_helper.compute_robot_pcd(
                    curr_qpos, link_names=None, num_pts=None, pcd_name="finger"
                )
                pcd_o3d = np2o3d(pcd)
                if i == 0:
                    visualizer = o3d.visualization.Visualizer()
                    visualizer.create_window()
                    curr_pcd = copy.deepcopy(pcd_o3d)
                    visualizer.add_geometry(curr_pcd)
                    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    visualizer.add_geometry(origin)

                curr_pcd.points = pcd_o3d.points
                curr_pcd.colors = pcd_o3d.colors

                start_time = time.time()
                while time.time() - start_time < 0.05:
                    visualizer.update_geometry(curr_pcd)
                    visualizer.poll_events()
                    visualizer.update_renderer()
                    if i == 0:
                        visualizer.run()


if __name__ == "__main__":
    main()
