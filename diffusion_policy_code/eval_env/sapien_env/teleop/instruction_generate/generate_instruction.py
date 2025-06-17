import numpy as np


class instruction_generater:
    def __init__(self, seed, keys, template_path, slackness_type) -> None:
        import random

        self.random = random
        self.random.seed(seed)
        self.keys = keys
        self.templates = self.load_templates(template_path, slackness_type)

    def load_templates(self, file_path, slackness_type):
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get(slackness_type, [])

    def fill_in_template(self, element):
        template = self.random.sample(self.templates, k=1)[0]
        instruction = template.format(**element)
        return instruction

    def get_absolute_position(self, layout, tgt_mug_idx):
        element_list = []
        other_element_list = []
        x = layout[:, 0]
        y = layout[:, 1]
        x_min_idx = np.argmin(x)
        if x_min_idx == tgt_mug_idx:
            element_list.append("left mug")
            other_element_list.append("right mug")
        x_max_idx = np.argmax(y)
        if x_max_idx == tgt_mug_idx:
            element_list.append("right mug")
            other_element_list.append("left mug")
        y_min_idx = np.argmin(x)
        if y_min_idx == tgt_mug_idx:
            element_list.append("back mug")
            other_element_list.append("front mug")
        y_max_idx = np.argmax(y)
        if y_max_idx == tgt_mug_idx:
            element_list.append("front mug")
            other_element_list.append("back mug")
        if len(element_list) != 0:
            sample_idx = self.random.sample(range(len(element_list)), k=1)[0]
            return [element_list[sample_idx], other_element_list[sample_idx]]
        else:
            return None

    def get_relative_position(self, layout, reference, tgt_branch_idx):
        dist = np.linalg.norm(layout - reference, ord=2, axis=-1)
        dist_max_idx = np.argmax(dist)
        if dist_max_idx == tgt_branch_idx:
            return ["furthest", "nearest"]
        else:
            return None

    def config_to_descriptive_element(
        self, config, mug_list, vis_info, blank_descriptive_element
    ):
        mug_idx = config["init"]
        branch_idx = config["tgt"]
        mug_layout = config["init_layout"]
        branch_layout = config["tgt_layout"]
        assigned_mug_position = mug_layout[mug_idx]
        mug_name = mug_list[mug_idx[0]]
        assigned_mug_vis = vis_info["mug"][mug_name]
        assigned_branch_name = vis_info["branch"][str(branch_idx[0])]
        blank_descriptive_element["mug"]["color"] = assigned_mug_vis
        blank_descriptive_element["mug"]["position"] = self.get_absolute_position(
            mug_layout, mug_idx[0]
        )
        blank_descriptive_element["branch"]["position"] = assigned_branch_name
        blank_descriptive_element["branch"]["relative_position"] = (
            self.get_relative_position(
                branch_layout, assigned_mug_position, branch_idx[0]
            )
        )
        return blank_descriptive_element

    def distill_descriptive_element(self, descriptive_element):
        distill_element = {}
        for key in descriptive_element.keys():
            descriptive = descriptive_element[key]
            element = []
            for sub_key in descriptive.keys():
                if descriptive[sub_key] is not None:
                    element.append(descriptive[sub_key])
            sel_element = self.random.sample(element, k=1)
            distill_element[key] = sel_element[0]

        distill_element["other_mug"] = distill_element["mug"][-1]
        distill_element["mug"] = distill_element["mug"][0]
        distill_element["other_branch"] = distill_element["branch"][-1]
        distill_element["branch"] = distill_element["branch"][0]
        return distill_element

    def element_to_str(self, input_element):
        result = ", ".join([f"{key}: {value}" for key, value in input_element.items()])
        return result

    def from_config_to_string(self, config, mug_list, vis_info):
        blank_descriptive_element = {
            "mug": {
                "color": None,
                "position": None,
            },
            "branch": {
                "position": None,
                "relative_position": None,
            },
        }

        descriptive_element = self.config_to_descriptive_element(
            config=config,
            mug_list=mug_list,
            vis_info=vis_info,
            blank_descriptive_element=blank_descriptive_element,
        )

        descriptive_element = self.distill_descriptive_element(descriptive_element)
        return descriptive_element


if __name__ == "__main__":
    mug_list = ["nescafe1", "nescafe2", "nescafe3"]

    config = {
        "init": np.array([0]),
        "init_layout": np.array(
            [
                [-0.07552315, -0.13899174, 0.04233006],
                [0.01711246, -0.00962532, 0.04233007],
                [0.21464348, -0.06992011, 0.04233007],
            ],
            dtype=np.float32,
        ),
        "tgt": np.array([2]),
        "tgt_layout": np.array(
            [
                [0.04604891, 0.15104604, 0.36848236],
                [-0.05060494, 0.15107961, 0.3704836],
                [-0.02316604, 0.18560973, 0.26565792],
            ]
        ),
    }

    slackness_type = "branch_slackness"

    generater = instruction_generater(
        seed=4,
        keys=["mug", "branch"],
        template_path="/home/neo/lyt_dp/diffusion/sapien_env/sapien_env/teleop/instruction_generate/template.json",
        slackness_type=slackness_type,
    )
    descriptive_element = generater.from_config_to_string(config, mug_list)
    instruction = generater.fill_in_template(descriptive_element)
    print(instruction)
