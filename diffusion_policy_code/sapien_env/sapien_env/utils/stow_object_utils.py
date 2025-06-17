from pathlib import Path
import os

import numpy as np
import sapien.core as sapien

DEFAULT_SCALE = {
    "shelf": 0.07,
    "env_book_0": 0.005,
    "env_book_1": 0.005,
    "book_1": 0.005,
    "book_2": 0.02,
    "flakes_1": 0.008,
    "flakes_2": 0.006,
}

DEFAULT_DENSITY = {
    "shelf": 1000,
    "book_1": 100,
    "book_2": 100,
    "flakes_1": 100,
    "flakes_2": 100,
}


def load_shelf(scene: sapien.Scene, scale=None, material=None, density=None):
    current_dir = Path(__file__).parent
    model_dir = current_dir.parent / "assets" / "stow"

    builder = scene.create_actor_builder()

    object_name = "shelf"
    visual_file = model_dir / object_name / f"{object_name}.glb"
    collision_file = visual_file

    if scale is None:
        scale = DEFAULT_SCALE[object_name]
    if isinstance(scale, float):
        scales = np.array([scale] * 3)
    elif isinstance(scale, list):
        scales = np.array(scale)
    else:
        print("scale must be float or list")
        raise NotImplementedError
    
    scales = scales * np.array([1.,1.,1.0])

    if density is None:
        density = DEFAULT_DENSITY.get(object_name)
    builder.add_visual_from_file(str(visual_file), scale=scales)
    builder.add_nonconvex_collision_from_file(
        str(collision_file),
        scale=scales,
        material=material,
    )

    shelf = builder.build_static(name="shelf")
    return shelf

def load_platform(scene: sapien.Scene, scale=None, material=None, density=None):
    current_dir = Path(__file__).parent
    model_dir = current_dir.parent / "assets" / "stow"

    builder = scene.create_actor_builder()

    object_name = "mug"
    visual_file = model_dir / object_name / f"{object_name}.glb"
    collision_file = visual_file


    builder.add_visual_from_file("/home/sim/general_dp-neo-attention_map/sapien_env/sapien_env/assets/yx/mug_2/mug_2.glb", scale=np.array([0.005,0.005,0.005]))
    builder.add_nonconvex_collision_from_file(
        "/home/sim/general_dp-neo-attention_map/sapien_env/sapien_env/assets/yx/mug_2/mug_2.glb",
        scale=np.array([0.005,0.005,0.005]),
        material=material,
    )

    shelf = builder.build(name="try")
    return shelf


def load_stow_obj(
    scene: sapien.Scene,
    object_name: str,
    name: str,
    scale=None,
    material=None,
    collision_shape="convex",
    density=None,
    is_static=False,
):
    current_dir = Path(__file__).parent
    model_dir = current_dir.parent / "assets" / "stow"

    visual_file = model_dir / object_name / f"{object_name}.glb"
    collision_file_cands = [
        model_dir / object_name / f"decomp.obj",
        model_dir / object_name / f"{object_name}_collision.obj",
        model_dir / object_name / f"{object_name}.glb",
    ]

    builder = scene.create_actor_builder()

    if scale is None:
        scale = DEFAULT_SCALE[object_name]
        scales = np.array([scale] * 3)
    else:
        scales = DEFAULT_SCALE[object_name] * scale
    
    #scales = scales * np.array([0.5,1,1])

    if density is None:
        density = DEFAULT_DENSITY.get(object_name)
    builder.add_visual_from_file(str(visual_file), scale=scales)

    for collision_file in collision_file_cands:
        if os.path.exists(str(collision_file)):
            if collision_shape == "convex":
                builder.add_collision_from_file(
                    str(collision_file),
                    scale=scales,
                    material=material,
                    density=density,
                )
            elif collision_shape == "nonconvex":
                builder.add_nonconvex_collision_from_file(
                    str(collision_file),
                    scale=scales,
                    material=material,
                )
            elif collision_shape == "multiple":
                builder.add_multiple_collisions_from_file(
                    str(collision_file),
                    scale=scales,
                    material=material,
                    density=density,
                )
            break

    actor = (
        builder.build_static(name=name)
        if is_static
        else builder.build(name=name)
    )
    return actor
