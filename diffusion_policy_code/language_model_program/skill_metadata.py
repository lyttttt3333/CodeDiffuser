import sys
import os
import random
from pathlib import Path
import glob
from omegaconf import OmegaConf
import hydra
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dotenv import load_dotenv
from openai import OpenAI
import cv2
import yaml

from utils import bcolors, encode_image
from sapien_env.utils.render_scene_utils import project_points

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(version_base=None)
def main(cfg: OmegaConf):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    assert (
        api_key is not None
    ), "Please set OPENAI_API_KEY in your environment variables"
    client = OpenAI(api_key=api_key)

    OmegaConf.resolve(cfg)
    assert cfg.is_real == False, "This script is for simulation only"
    assert cfg.input_type in ["plot", "video"], "Invalid input type"
    if hasattr(cfg.task, "dataset_dir"):
        dataset_dir = cfg.task.dataset_dir
        sys.path.append(dataset_dir)
    dataset_dir = Path(dataset_dir).expanduser()
    episodes_paths = glob.glob(os.path.join(dataset_dir, "episode_*.hdf5"))
    episodes_stem_name = [Path(path).stem for path in episodes_paths]
    episodes_idx = [int(stem_name.split("_")[-1]) for stem_name in episodes_stem_name]
    episodes_idx = sorted(episodes_idx)
    assert episodes_idx, "No episodes found in the dataset directory"

    epi_idx = random.choice(episodes_idx)
    dataset_path = os.path.join(dataset_dir, f"episode_{epi_idx}.hdf5")
    with h5py.File(dataset_path) as file:
        curr_repo_dir = os.path.abspath(__file__ + "/../../")
        save_skill_path = os.path.join(curr_repo_dir, f"data/outputs/{cfg.skill_name}")
        save_dataset_path = os.path.join(save_skill_path, cfg.dataset_name)
        Path(save_dataset_path).mkdir(parents=True, exist_ok=True)

        if cfg.input_type == "plot":
            meta_image = file["observations"]["meta_images"]["color"][0]
            image_path = os.path.join(save_dataset_path, f"episode_{epi_idx}.png")

            print(f"{bcolors.OKGREEN}Generating image at: {image_path}{bcolors.ENDC}")
            print(
                f"{bcolors.OKGREEN}   - resolution: {meta_image.shape[1]}x{meta_image.shape[0]}{bcolors.ENDC}"
            )

            trajectory = np.array(file["cartesian_action"])
            intrinsic = np.array(file["observations"]["meta_images"]["intrinsic"][0])
            extrinsic = np.array(file["observations"]["meta_images"]["extrinsic"][0])[
                :-1
            ]
            points = project_points(trajectory[:, :3], intrinsic, extrinsic)

            plt.imshow(meta_image)
            plt.axis("off")
            plt.scatter(
                points[:, 0],
                points[:, 1],
                c=range(points.shape[0]),
                cmap="viridis",
                s=10,
            )
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            cbar = plt.colorbar(cax=cax)
            cbar.ax.tick_params(labelsize=6)
            plt.tight_layout(pad=0)
            if cfg.vis:
                plt.show()
            plt.savefig(image_path, dpi=600)
            plt.clf()

            meta_image = cv2.cvtColor(meta_image, cv2.COLOR_BGR2RGB)
            _, meta_image = cv2.imencode(".png", meta_image)
            base64_image = encode_image(meta_image)
            PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's in this image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "low",
                            },
                        },
                    ],
                },
            ]
            params = {
                "model": "gpt-4o",
                "messages": PROMPT_MESSAGES,
                "max_tokens": 300,
            }
        elif cfg.input_type == "video":
            frame_step = 5
            meta_images = file["observations"]["meta_images"]["color"]
            meta_images = meta_images[::frame_step]
            video_path = os.path.join(save_dataset_path, f"episode_{epi_idx}.mp4")

            print(f"{bcolors.OKGREEN}Generating video at: {video_path}{bcolors.ENDC}")
            print(
                f"{bcolors.OKGREEN}   - # of frames: {len(meta_images)}{bcolors.ENDC}"
            )
            print(
                f"{bcolors.OKGREEN}   - resolution: {meta_images[0].shape[1]}x{meta_images[0].shape[0]}{bcolors.ENDC}"
            )

            fps = 30
            output_size = (meta_images[0].shape[1], meta_images[0].shape[0])
            video = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                output_size,
            )
            for idx, image in enumerate(meta_images):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.putText(
                    image,
                    f"Frame {idx}/{len(meta_images)}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                if cfg.vis:
                    cv2.imshow("Skill Video", image)
                    key = cv2.waitKey(frame_step * 30)
                for _ in range(frame_step):
                    video.write(image)
            if cfg.vis:
                cv2.destroyAllWindows()
            video.release()

            meta_image_frames = []
            for image in meta_images:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                _, image = cv2.imencode(".png", image)
                base64_image = encode_image(image)
                meta_image_frames.append(base64_image)

            PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "These are frames from a video that I want to upload. Generate a list of the task-related objects in the video, a description of the task, and action steps of the robot arm with gripper in the video. Also, please provide a ending condition for this task.",
                        },
                        *map(
                            lambda x: {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{x}",
                                    "detail": "low",
                                },
                            },
                            meta_image_frames,
                        ),
                    ],
                },
            ]
            params = {
                "model": "gpt-4o",
                "messages": PROMPT_MESSAGES,
                "max_tokens": 300,
            }

        response = client.chat.completions.create(**params)
        message = response.choices[0].message.content

        term_size = os.get_terminal_size()
        print(f"{bcolors.OKBLUE}-{bcolors.ENDC}" * term_size.columns)
        print(f"\n{bcolors.OKCYAN}{message}{bcolors.ENDC}\n")
        term_size = os.get_terminal_size()
        print(f"{bcolors.OKBLUE}-{bcolors.ENDC}" * term_size.columns)

        print(
            f"{bcolors.WARNING}Token Usage: {response.usage.total_tokens}{bcolors.ENDC}"
        )
        print(
            f"{bcolors.WARNING}   - prompt_tokens: {response.usage.prompt_tokens}{bcolors.ENDC}"
        )
        print(
            f"{bcolors.WARNING}   - completion_tokens: {response.usage.completion_tokens}{bcolors.ENDC}"
        )
        skill_metadata = {
            "skill_name": cfg.skill_name,
            "description": message,
        }

        yaml_path = os.path.join(save_skill_path, f"metadata.yaml")
        print(f"{bcolors.OKGREEN}Saving metadata at: {yaml_path}{bcolors.ENDC}")
        with open(yaml_path, "w+") as file:
            # metadata_lib = yaml.safe_load(file)
            yaml.dump(skill_metadata, file)


if __name__ == "__main__":
    main()
