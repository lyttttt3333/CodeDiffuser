from diffusion_policy.common.replay_buffer import ReplayBuffer
import zarr
import numpy as np
import os

if True:
    dir_name = "merge_pcd_540"

    cache_zarr_path_list = ["/home/yitong/diffusion/data_train/hang_mug_360/pcd_000_180.zarr.zip",
                            "/home/yitong/diffusion/data_train/hang_mug_360/pcd_180_360.zarr.zip",
                            "/home/yitong/diffusion/data_train/hang_mug_360/pcd_360_540.zarr.zip"]

    sel_num_list = [180, 180,180]
    cache_zarr_path_to_save_root = "/home/yitong/diffusion/data_train"
    cache_zarr_path_to_save_dir = os.path.join(cache_zarr_path_to_save_root,dir_name)
    if not os.path.exists(cache_zarr_path_to_save_dir):
        os.mkdir(cache_zarr_path_to_save_dir)

    root = zarr.group(zarr.MemoryStore())
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    action_list = list()
    d3fields_list = list()
    embedding_list = list()
    episode_ends_list = list()

    abs_last_part_end = 0
    for idx, cache_zarr_path in enumerate(cache_zarr_path_list):
        with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
            replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store, store=zarr.MemoryStore()
            )
        a = replay_buffer.keys()
        sel_num = sel_num_list[idx]


        episode_ends = replay_buffer.meta["episode_ends"][:sel_num] + abs_last_part_end
        abs_last_part_end = episode_ends[-1]

        sel_end = replay_buffer.meta["episode_ends"][:sel_num][-1]

        action = replay_buffer["action"][:sel_end]
        d3fields = replay_buffer["d3fields"][:sel_end]
        embedding = replay_buffer["embedding"][:sel_end]

        action_list.append(action)
        d3fields_list.append(d3fields)
        embedding_list.append(embedding)
        episode_ends_list.append(episode_ends)

        del replay_buffer

    action_list = np.concatenate(action_list, axis=0)
    d3fields_list = np.concatenate(d3fields_list, axis=0)
    embedding_list = np.concatenate(embedding_list, axis=0)
    episode_ends_list = np.concatenate(episode_ends_list, axis=-1)

    _ = data_group.array(
        name="d3fields",
        data=d3fields_list,
        shape=d3fields_list.shape,
        chunks= (1,) + d3fields_list.shape[1:],
        compressor=None,
        dtype=d3fields_list.dtype,
    )

    _ = data_group.array(
        name="action",
        data=action_list,
        shape=action_list.shape,
        chunks=action_list.shape,
        compressor=None,
        dtype=action_list.dtype,
    )

    _ = data_group.array(
        name="embedding",
        data=embedding_list,
        shape=embedding_list.shape,
        chunks=embedding_list.shape,
        compressor=None,
        dtype=embedding_list.dtype,
    )


    _ = meta_group.array(
        "episode_ends", episode_ends_list, dtype=np.int64, compressor=None, overwrite=True
    )

    merged_replay_buffer = ReplayBuffer(root)

    cache_zarr_path_to_save = os.path.join(cache_zarr_path_to_save_dir, "cache_no_seg_no_dino_eef.zarr.zip")
    with zarr.ZipStore(cache_zarr_path_to_save) as zip_store:
        merged_replay_buffer.save_to_store(store=zip_store)


