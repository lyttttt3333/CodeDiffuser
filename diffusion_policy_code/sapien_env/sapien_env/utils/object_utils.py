from pathlib import Path
import os
import transforms3d
import numpy as np
import sapien.core as sapien


def load_open_box(
    scene: sapien.Scene,
    renderer: sapien.SapienRenderer,
    half_l,
    half_w,
    h,
    floor_width,
    origin=[0, 0, 0],
    color=None,
):
    if not color:
        asset_dir = Path(__file__).parent.parent / "assets"
        map_path = asset_dir / "misc" / "dark-wood.png"
        box_visual_material = renderer.create_material()
        box_visual_material.set_metallic(0.0)
        box_visual_material.set_specular(0.3)
        box_visual_material.set_diffuse_texture_from_file(str(map_path))
        box_visual_material.set_roughness(0.3)

    builder = scene.create_actor_builder()

    # bottom
    box_bottom_origin = origin.copy()
    box_bottom_origin[2] += floor_width / 2
    builder.add_box_collision(
        half_size=[half_l, half_w, floor_width / 2],
        pose=sapien.Pose(p=box_bottom_origin),
    )
    if color:
        builder.add_box_visual(
            half_size=[half_l, half_w, floor_width / 2],
            color=color,
            pose=sapien.Pose(p=box_bottom_origin),
        )
    else:
        builder.add_box_visual(
            half_size=[half_l, half_w, floor_width / 2],
            material=box_visual_material,
            pose=sapien.Pose(p=box_bottom_origin),
        )

    # left
    box_left_origin = origin.copy()
    box_left_origin[0] += -half_l
    box_left_origin[2] += h / 2
    builder.add_box_collision(
        half_size=[floor_width / 2, half_w + floor_width / 2, h / 2],
        pose=sapien.Pose(p=box_left_origin),
    )
    if color:
        builder.add_box_visual(
            half_size=[floor_width / 2, half_w + floor_width / 2, h / 2],
            color=color,
            pose=sapien.Pose(p=box_left_origin),
        )
    else:
        builder.add_box_visual(
            half_size=[floor_width / 2, half_w + floor_width / 2, h / 2],
            material=box_visual_material,
            pose=sapien.Pose(p=box_left_origin),
        )

    # right
    box_right_origin = origin.copy()
    box_right_origin[0] += half_l
    box_right_origin[2] += h / 2
    builder.add_box_collision(
        half_size=[floor_width / 2, half_w + floor_width / 2, h / 2],
        pose=sapien.Pose(p=box_right_origin),
    )
    if color:
        builder.add_box_visual(
            half_size=[floor_width / 2, half_w + floor_width / 2, h / 2],
            color=color,
            pose=sapien.Pose(p=box_right_origin),
        )
    else:
        builder.add_box_visual(
            half_size=[floor_width / 2, half_w + floor_width / 2, h / 2],
            material=box_visual_material,
            pose=sapien.Pose(p=box_right_origin),
        )

    # back
    box_back_origin = origin.copy()
    box_back_origin[1] += -half_w
    box_back_origin[2] += h / 2
    builder.add_box_collision(
        half_size=[half_l + floor_width / 2, floor_width / 2, h / 2],
        pose=sapien.Pose(p=box_back_origin),
    )
    if color:
        builder.add_box_visual(
            half_size=[half_l + floor_width / 2, floor_width / 2, h / 2],
            color=color,
            pose=sapien.Pose(p=box_back_origin),
        )
    else:
        builder.add_box_visual(
            half_size=[half_l + floor_width / 2, floor_width / 2, h / 2],
            material=box_visual_material,
            pose=sapien.Pose(p=box_back_origin),
        )

    # front
    box_front_origin = origin.copy()
    box_front_origin[1] += half_w
    box_front_origin[2] += h / 2
    builder.add_box_collision(
        half_size=[half_l, floor_width / 2, h / 2], pose=sapien.Pose(p=box_front_origin)
    )
    if color:
        builder.add_box_visual(
            half_size=[half_l, floor_width / 2, h / 2],
            color=color,
            pose=sapien.Pose(p=box_front_origin),
        )
    else:
        builder.add_box_visual(
            half_size=[half_l, floor_width / 2, h / 2],
            material=box_visual_material,
            pose=sapien.Pose(p=box_front_origin),
        )

    box = builder.build_static(name="open box")
    return box


DEFAULT_SCALE = {
    "cola": 0.37,
    "pepsi": 1.3,
    "mtndew": 0.025,
    "sprite": 0.3,
    "obamna": 0.0003,
    "diet_soda": 0.008,
    "drpepper": 0.04,
    "white_pepsi": 0.016,
    "mug_tree": 1.1,
    "headless_mug_tree": 1.1,
    "nescafe_mug_1": [0.011, 0.011, 0.0105],
    "nescafe_mug_2": [0.011, 0.011, 0.0105],
    "nescafe_mug_3": [0.011, 0.011, 0.0105],
    "nescafe_mug_4": [0.011, 0.011, 0.0105],
    "aluminum_mug": 1.3,
    "beer_mug": 0.0006,
    "black_mug": 0.175,
    "white_mug": 0.06,
    "blue_mug": 1.0,
    "kor_mug": 0.8,
    "low_poly_mug": 0.05,
    "blender_mug": 0.007,
    "mug_2": 0.006,
    "pencil": 0.05,
    "sharpener": 0.02,
    "sharpener_2": 0.02,
    "pencil_2": 0.15,
    "pencil_3": 0.03,
    "pencil_4": 3.0,
    "pencil_5": 0.01,
    "battery_container": 0.01,
    "battery_1": [0.09, 0.1, 0.09],
    "battery_2": [0.09, 0.1, 0.09],
    "battery_3": [0.09, 0.1, 0.09],
    "battery_4": [0.09, 0.1, 0.09],
    "battery_5": [0.09, 0.1, 0.09],
}

DEFAULT_DENSITY = {
    "aluminum_mug": 100,
    "blue_mug": 1,
}

def load_platform(
    scene: sapien.Scene,
    object_name,
    scale=None,
    material=None,
    collision_shape="convex",
    density=None,
    is_static=False,
    pose=np.array([np.pi/2,0,0]),
):
    current_dir = Path(__file__).parent
    yx_dir = current_dir.parent / "assets" / "yx"

    visual_file = yx_dir / object_name / f"{object_name}.glb"
    collision_file_cands = [
        yx_dir / object_name / f"decomp.obj",
        yx_dir / object_name / f"{object_name}_collision.obj",
        yx_dir / object_name / f"{object_name}.glb",
    ]

    builder = scene.create_actor_builder()

    if scale is None:
        scales = np.array([1.1,1.1,1.1])
    if False:
        if isinstance(scale, float):
            scales = np.array([scale] * 3)
        elif isinstance(scale, list):
            scales = np.array(scale)
        else:
            print("scale must be float or list")
            raise NotImplementedError

    if density is None:
        density = DEFAULT_DENSITY.get(object_name, 10)
    
    pose = transforms3d.euler.euler2quat(3.14/2,0,3.14, axes="sxyz")
    pose = sapien.Pose(p=np.array([0.,0.15,0.]),q=pose)
    builder.add_visual_from_file(str(visual_file), scale=scales,pose=pose)

    

    if object_name == "pencil":
        builder.add_box_collision(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[0.01, 0.01, 0.08],
            density=density,
            material=material,
        )
    elif object_name == "pencil_2":
        builder.add_box_collision(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[0.01, 0.01, 0.09],
            density=density,
            material=material,
        )
    elif object_name == "pencil_3":
        builder.add_box_collision(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[0.01, 0.01, 0.12],
            density=density,
            material=material,
        )
    elif object_name == "pencil_4":
        builder.add_box_collision(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[0.01, 0.01, 0.06],
            density=density,
            material=material,
        )
    elif object_name == "pencil_5":
        builder.add_box_collision(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[0.012, 0.012, 0.10],
            density=density,
            material=material,
        )
    else:
        for collision_file in collision_file_cands:
            if os.path.exists(str(collision_file)):
                if collision_shape == "convex":
                    builder.add_collision_from_file(
                        str(collision_file),
                        scale=scales,
                        material=material,
                        density=density,
                        pose=pose,
                    )
                elif collision_shape == "nonconvex":
                    builder.add_nonconvex_collision_from_file(
                        str(collision_file),
                        scale=scales,
                        material=material,
                        #density=density,
                        pose=pose,
                    )
                elif collision_shape == "multiple":
                    builder.add_multiple_collisions_from_file(
                        str(collision_file),
                        scale=scales,
                        material=material,
                        density=density,
                        pose=pose,
                    )
                break

    if is_static:
        actor = builder.build_static(name=object_name)
    else:
        actor = builder.build(name=object_name)
    return actor

def load_obj(
    scene: sapien.Scene,
    object_name,
    scale=None,
    material=None,
    collision_shape="convex",
    density=None,
    is_static=False,
    pose= None
):
    current_dir = Path(__file__).parent
    yx_dir = current_dir.parent / "assets" / "yx"

    visual_file = yx_dir / object_name / f"{object_name}.glb"
    collision_file_cands = [
        yx_dir / object_name / f"decomp.obj",
        yx_dir / object_name / f"{object_name}_collision.obj",
        yx_dir / object_name / f"{object_name}.glb",
    ]

    builder = scene.create_actor_builder()

    if scale is None:
        scale = DEFAULT_SCALE[object_name]
    if isinstance(scale, float):
        scales = np.array([scale] * 3)
    elif isinstance(scale, list):
        scales = np.array(scale)
    else:
        print("scale must be float or list")
        raise NotImplementedError

    if density is None:
        density = DEFAULT_DENSITY.get(object_name, 10)

    if pose is None:
        builder.add_visual_from_file(str(visual_file), scale=scales)
    else:
        builder.add_visual_from_file(str(visual_file), scale=scales, pose=pose)


    if object_name == "pencil":
        builder.add_box_collision(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[0.01, 0.01, 0.08],
            density=density,
            material=material,
        )
    elif object_name == "pencil_2":
        builder.add_box_collision(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[0.01, 0.01, 0.09],
            density=density,
            material=material,
        )
    elif object_name == "pencil_3":
        builder.add_box_collision(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[0.01, 0.01, 0.12],
            density=density,
            material=material,
        )
    elif object_name == "pencil_4":
        builder.add_box_collision(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[0.01, 0.01, 0.06],
            density=density,
            material=material,
        )
    elif object_name == "pencil_5":
        builder.add_box_collision(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[0.012, 0.012, 0.10],
            density=density,
            material=material,
        )
    else:
        for collision_file in collision_file_cands:
            if os.path.exists(str(collision_file)):
                if pose is None:
                    if collision_shape == "convex":
                        builder.add_collision_from_file(
                            str(collision_file),
                            scale=scales,
                            material=material,
                            density=density,
                            #pose=pose
                        )
                    elif collision_shape == "nonconvex":
                        builder.add_nonconvex_collision_from_file(
                            str(collision_file),
                            scale=scales,
                            material=material,
                            #density=density,
                            #pose=pose
                        )
                    elif collision_shape == "multiple":
                        builder.add_multiple_collisions_from_file(
                            str(collision_file),
                            scale=scales,
                            material=material,
                            density=density,
                            #pose=pose
                        )
                else:
                    if collision_shape == "convex":
                        builder.add_collision_from_file(
                            str(collision_file),
                            scale=scales,
                            material=material,
                            density=density,
                            pose=pose
                        )
                    elif collision_shape == "nonconvex":
                        builder.add_nonconvex_collision_from_file(
                            str(collision_file),
                            scale=scales,
                            material=material,
                            #density=density,
                            pose=pose
                        )
                    elif collision_shape == "multiple":
                        builder.add_multiple_collisions_from_file(
                            str(collision_file),
                            scale=scales,
                            material=material,
                            density=density,
                            pose=pose
                        )
                break

    if is_static:
        actor = builder.build_static(name=object_name)
    else:
        actor = builder.build(name=object_name)
    return actor
