


import os
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'
import h5py
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.transform import Rotation as R
from d3fields.fusion import Fusion, create_init_grid
from sklearn.cluster import DBSCAN  
from sklearn.preprocessing import StandardScaler  
from d3fields.utils.my_utils import fps_np
import os

# from diffusion_policy.common.data_utils import load_dict_from_hdf5
# from d3fields.utils.draw_utils import (
#     draw_keypoints,
#     o3dVisualizer,
# )

def load_dict_from_hdf5(filename):
    """
    ....
    """
    # with h5py.File(filename, 'r') as h5file:
    #     return recursively_load_dict_contents_from_group(h5file, '/')
    h5file = h5py.File(filename, "r")
    return recursively_load_dict_contents_from_group(h5file, "/"), h5file


def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            # ans[key] = np.array(item)
            ans[key] = item
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + "/"
            )
    return ans

def extract_dinov2_feats(imgs, model, device = "cuda"):
    dtype = torch.float16
    K, H, W, _ = imgs.shape

    patch_h = H // 20
    patch_w = W // 20
    feat_dim = 1024

    transform = T.Compose(
        [
            T.Resize((patch_h * 14, patch_w * 14)),
            T.CenterCrop((patch_h * 14, patch_w * 14)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    imgs_tensor = torch.zeros(
        (K, 3, patch_h * 14, patch_w * 14), device=device, dtype=dtype
    )
    for j in range(K):
        img = Image.fromarray(imgs[j])
        imgs_tensor[j] = transform(img)[:3]
    with torch.no_grad():
        features_dict = model.forward_features(imgs_tensor.to(dtype=dtype))
        features = features_dict["x_norm_patchtokens"]
        features = features.reshape((K, patch_h, patch_w, feat_dim))
    features = features.permute(0, 3, 1, 2)
    features = F.interpolate(
        features, size=(H, W), mode="bilinear", align_corners=False
    )
    features = features.permute(0, 2, 3, 1)
    return features

def query_point_feat(x, y, dinov2_model, src_img, device):

    # path = "/home/sim/general_dp-neo-attention_map/fields/hang_mug/color_0.png"
    # src_img = cv2.imread(path)[..., ::-1]
 
    src_feats = extract_dinov2_feats(src_img[None], dinov2_model, device)[0]

    feats_h, feats_w = src_feats.shape[:2]
    img_h, img_w = src_img.shape[:2]
    src_feat_tensor = src_feats[
        int(y * feats_h / img_h), int(x * feats_w / img_w)
    ]
    return src_feat_tensor

def compute_similarity_tensor(src_feat_map, tgt_feat, pcd):
    assert src_feat_map.shape[1] == tgt_feat.shape[0]

    similarity = torch.nn.functional.cosine_similarity(
        src_feat_map, tgt_feat[None], dim=1
    )

    similarity = (similarity - torch.min(similarity))/(torch.max(similarity)-torch.min(similarity))
    k = min(60, similarity.shape[0])
    similar_value, similar_index = torch.topk(similarity, k) 

    similar_threshold = similar_value[-1]
    similarity = similarity /similar_threshold
    similarity[similarity>=1] = 1
    similarity = torch.pow(similarity, 3)

    sim_feat_map = src_feat_map[similar_index[:30]]
    sim_feat = torch.mean(sim_feat_map, dim=0).reshape(-1)

    indices = similar_index.detach().cpu().numpy().astype(np.int16)
    attention_pts = pcd[indices,:]

    return attention_pts, similarity.cpu().numpy().astype(np.float64)[:,None], sim_feat



def query_img_position(key_point, extrinsics, intrinsics, cam_list):
    # grasp_p = key_point[0]
    place_p = key_point
    # grasp_p = np.concatenate([grasp_p,np.array([1])],axis = -1)[:,None]
    place_p = np.concatenate([place_p,np.array([1])],axis = -1)[:,None]
    position_list = []
    for idx, cam in enumerate(cam_list):
        T = extrinsics[idx]
        K = intrinsics[idx]
        PC = np.dot(T, place_p)
        PC = PC[:3]/PC[3]
        PN = np.dot(K, PC[:3])
        x = int(PN[0]/PN[2])
        y = int(PN[1]/PN[2])
        position_list.append(np.array([[x,y]]))
    return np.concatenate(position_list,axis=0)

def check3d(key_point, bg):
    color = np.ones_like(key_point)*np.array([255,0,0])
    full_kp = np.concatenate([key_point, color],axis=-1)
    color = np.ones_like(bg)*np.array([0,0,255])
    full_bg = np.concatenate([bg, color],axis=-1)
    full = np.concatenate([full_kp,full_bg],axis=0)
    np.savetxt("/home/sim/general_dp-neo-attention_map/keypoint.txt",full)


def check2d(xys, imgs):
    for i in range(imgs.shape[0]):
        x = xys[i][0]
        y = xys[i][1]
        img = imgs[i]
        img = np.concatenate([img[:,:,2][:,:,None],img[:,:,1][:,:,None],img[:,:,0][:,:,None]],axis=-1)
        for x_i in [-3,-2,-1,0,1,2,3]:
            for y_i in [-3,-2,-1,0,1,2,3]:
                img[y + y_i, x + x_i,:]=np.array([0,255,0])
        cv2.imwrite(img=img,filename=f"/root/lyt/test_keypoint_{i}.png")
        a=0

def query_attach_pts(vertices, target_pts):
    rela_vertice = vertices - target_pts
    rela_dist = np.linalg.norm(rela_vertice,axis=-1)
    nearest_dist = np.min(rela_dist)
    threshold = nearest_dist * 2
    indice = (rela_dist <= threshold)
    near_vertices = vertices[indice,:]
    centroid = np.mean(near_vertices,axis=0)
    return centroid, indice
    

        


class Corr():
    def __init__(self, data_dict, vertice, t, fusion:Fusion, device):
        self.dtype = torch.float16
        self.step = 0.002
        self.t = t
        self.fusion = fusion
        self.device = device

        self.cam_name_list = [
            "left_bottom_view",
            "right_bottom_view",
            "left_top_view",
            "right_top_view",
        ]

        colors = []
        for i in self.cam_name_list:
            img = data_dict["observations"]["images"][f"{i}_color"][self.t]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            colors.append(img)
        self.colors = np.stack(colors, axis=0)

        self.depths = (
            np.stack(
                [
                    data_dict["observations"]["images"][f"{i}_depth"][self.t]
                    for i in self.cam_name_list
                ],
                axis=0,
            )
            / 1000.0
        )

        self.extrinsics = np.stack(
            [
                data_dict["observations"]["images"][f"{i}_extrinsic"][self.t]
                for i in self.cam_name_list
            ]
        )
        self.intrinsics = np.stack(
            [
                data_dict["observations"]["images"][f"{i}_intrinsic"][self.t]
                for i in self.cam_name_list
            ]
        )

        self.obs = {
            "color": self.colors,
            "depth": self.depths,
            "pose": self.extrinsics[:, :3],  # (N, 3, 4)
            "K": self.intrinsics,
        }

        self.fusion.update(self.obs)
        self.vertices = vertice

    def get_base_feat(self):
        vertices_tensor = torch.from_numpy(self.vertices).to(device, dtype=self.dtype)
        with torch.no_grad():
            out = fusion.batch_eval(
                vertices_tensor, return_names=["dino_feats","color_tensor"]
            )
        self.vertices_feats_tensor = out["dino_feats"]
        self.vertices_color = out["color_tensor"].cpu().numpy()
        
    def draw_corr(self, use_keypoint, use_feat, iterative = 1, key_point = None, fig_index = None, feat = None):
        if use_keypoint:
            assert key_point is not None
            assert fig_index is not None
            query_img_index = fig_index
            query_img = self.colors[query_img_index]
            key_point, _ = query_attach_pts(vertices=self.vertices, target_pts=key_point)
            img_positions = self.trans_3d_2d(key_point, all=False)
            src_feat_tensor = query_point_feat(x = img_positions[0], y = img_positions[1], dinov2_model=fusion.dinov2_feat_extractor, src_img=query_img, device=device)

        if use_feat:
            assert feat is not None
            src_feat_tensor = torch.from_numpy(feat).to(self.dtype).to(self.device)

        for _ in range(iterative):
            attention_pts, similarity, src_feat_tensor = compute_similarity_tensor(self.vertices_feats_tensor, src_feat_tensor, self.vertices) 
        return attention_pts, similarity, self.vertices, src_feat_tensor.detach().cpu().numpy().astype(np.float16)
    
    def trans_3d_2d(self, key_point, img_index = None):
        img_positions = query_img_position(key_point=key_point, extrinsics=self.extrinsics, intrinsics=self.intrinsics, cam_list=self.cam_name_list)
        if img_index is None:
            return img_positions
        else:
            return img_positions[img_index]
        
    def get_img(self, img_index):
        return self.colors[img_index]



def draw_attention(text, data_dict, fusion_model, device, seed = 0):
    attention_pts = draw_corr(data_dict=data_dict, fusion=fusion_model, device=device)

    X = StandardScaler().fit_transform(attention_pts)  
 
    db = DBSCAN(eps=0.3, min_samples=10).fit(X) 
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)  
    core_samples_mask[db.core_sample_indices_] = True  
    labels = db.labels_  
    
    unique_labels = set(labels)  
    attention_pts_list = list()
    for label in unique_labels:
        indice = (labels == label)
        attention_group = attention_pts[indice]
        attention_pts_list.append(attention_group)

    attention_pts_list = list()
    for i in range(6):
        path = f"/home/sim/general_dp-neo-attention_map/attention_pcd_{i}.txt"
        pts = np.loadtxt(path)
        attention_pts_list.append(pts)

    attention_part = query_text(attention_pts_list,text=text)



    pts_num = 100 if attention_part.shape[0] >= 100 else attention_part.shape[0]
    attention_part, _, _ = fps_np(attention_part, pts_num, seed = seed)
    return attention_part

    # np.savetxt("/home/sim/general_dp-neo-attention_map/attention_pcd.txt",attention_pts)
    # attention_pts = np.loadtxt("/home/sim/general_dp-neo-attention_map/attention_pcd.txt")

    # for idx, pts in enumerate(attention_pts_list):
    #     np.savetxt(f"/home/sim/general_dp-neo-attention_map/attention_pcd_{idx}.txt",pts)

    # attention_pts_list = list()
    # for i in range(6):
    #     path = f"/home/sim/general_dp-neo-attention_map/attention_pcd_{i}.txt"
    #     pts = np.loadtxt(path)
    #     attention_pts_list.append(pts)


    # if False:
    #     full_list = list()
    #     for i, attention_group in enumerate(attention_pts_list):
    #         color = np.ones_like(attention_group)
    #         if i == 0:
    #             color_piece = np.array([[135,206,235]])
    #         if i == 1:
    #             color_piece = np.array([[255,165,0]])
    #         if i == 2:
    #             color_piece = np.array([[124,256,0]])
    #         if i == 3:
    #             color_piece = np.array([[128,0,128]])
    #         if i == 4:
    #             color_piece = np.array([[255,255,0]])
    #         if i == 5:
    #             color_piece = np.array([[255,255,255]])
                
    #         color = color_piece * color
    #         full = np.concatenate([attention_group, color],axis=-1)
    #         full_list.append(full)
    #     full_list = np.concatenate(full_list, axis=0)
    #     np.savetxt("/home/sim/general_dp-neo-attention_map/attention_pcd_with_color.txt",full_list)
def query_text(attention_pts_list,text):
    attention_cent_list = list()
    for idx, attention_pts in enumerate(attention_pts_list):
        attention_cent = np.mean(attention_pts,axis=0)
        attention_cent_list.append(attention_cent[None,:])
    attention_cent_list = np.concatenate(attention_cent_list,axis=0)
    z = attention_cent_list[:,2]
    sort_indice = np.argsort(z)[::-1]
    top_2 = sort_indice[:2]
    middle_2 = sort_indice[2:4]
    if text == "the right top branch":
        potential_cent = attention_cent_list[top_2,:]
        if potential_cent[0,0] > potential_cent[1,0]:
            idx = top_2[0]
            return attention_pts_list[idx]
        else:
            idx = top_2[1]
            return attention_pts_list[idx]
    if text == "the left top branch":
        potential_cent = attention_cent_list[top_2,:]
        if potential_cent[0,0] > potential_cent[1,0]:
            idx = top_2[1]
            return attention_pts_list[idx]
        else:
            idx = top_2[0]
            return attention_pts_list[idx]
    if text == "the front middle branch":
        potential_cent = attention_cent_list[middle_2,:]
        if potential_cent[0,1] < potential_cent[1,1]:
            idx = middle_2[0]
            return attention_pts_list[idx]
        else:
            idx = middle_2[1]
            return attention_pts_list[idx]
        
def draw_fig(data_dict, vertice, fusion, device, feat_list, t, iterate_time, img_to_draw = 0):
    corr = Corr(data_dict, vertice, t, fusion, device)
    corr.get_base_feat()
    attention_pts_list = list()
    attention_cent_list = list()
    attention_cent_2d = list()
    next_feat_list = list()
    for _, feat in enumerate(feat_list):
        attention_pts, similarity, vertices, next_feat = corr.draw_corr(use_keypoint=False, use_feat=True, iterative=iterate_time, feat=feat)
        next_feat_list.append(next_feat)

        X = StandardScaler().fit_transform(attention_pts)  
    
        db = DBSCAN(eps=0.5, min_samples=10).fit(X) 
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)  
        core_samples_mask[db.core_sample_indices_] = True  
        labels = db.labels_  
        
        unique_labels = set(labels)  

        for _, label in enumerate(unique_labels):
            indice = (labels == label)
            attention_group = attention_pts[indice]
            group_num = attention_group.shape[0]
            if group_num == 0:
                continue
            else:
                attention_cent = np.mean(attention_group, axis=0)
                pos_on_fig = corr.trans_3d_2d(attention_cent, img_to_draw)
                attention_pts_list.append(attention_group)
                attention_cent_list.append(attention_cent)
                attention_cent_2d.append([pos_on_fig[0],pos_on_fig[1]])

    list_after_check=check_adj(pts_list=attention_pts_list,
              cent_list=attention_cent_list,
              cent_2d_list=attention_cent_2d)
    attention_pts_list,attention_cent_list,attention_cent_2d = list_after_check

    for i in range(len(attention_pts_list)):
        attention_pts = attention_pts_list[i]
        path = f"/root/lyt/test_attention_group/attention_group_{i}.txt"
        np.savetxt(path, attention_pts)

    # vis_pcd([corr.vertices]+attention_pts_list)

    base_image = Image.fromarray(corr.get_img(img_to_draw), 'RGB')  
    draw(base_image, attention_cent_2d)

    vertices = corr.vertices
    color = corr.vertices_color
    np.savetxt("vertice.txt",vertices)
    np.savetxt("vertice_color.txt",color)

    similarity_pcd = np.concatenate([vertices,similarity],axis=-1)

    return attention_pts_list, next_feat_list, similarity_pcd

def check_adj(pts_list, cent_list, cent_2d_list):

    cent_mat = np.concatenate(cent_list, axis=-1)
    cent_mat = cent_mat.reshape(-1,3)
    start_list, end_list = find_close_pairs(cent_mat, r = 0.05)
    new_pts_list = list()
    new_cent_list = list()
    new_cent_2d_list = list()
    for i in range(len(pts_list)):
        if i in start_list:
            index = start_list.index(i)
            start_index = start_list[index]
            end_index = end_list[index]
            start_pts = pts_list[start_index]
            end_pts = pts_list[end_index]
            pts = np.concatenate([start_pts, end_pts], axis=0)
            new_pts_list.append(pts)
            new_cent_list.append(np.mean(pts,axis=0))
            fig_x = (cent_2d_list[start_index][0]+cent_2d_list[end_index][0])/2
            fig_y = (cent_2d_list[start_index][1]+cent_2d_list[end_index][1])/2
            new_cent_2d_list.append([int(fig_x), int(fig_y)])
        elif i in end_list:
            continue
        else:
            new_pts_list.append(pts_list[i])
            new_cent_list.append(cent_list[i])
            new_cent_2d_list.append(cent_2d_list[i])
    return new_pts_list, new_cent_list, new_cent_2d_list

def find_close_pairs(points, r):   
    n = points.shape[0]  
    distances = np.sqrt(((points[:, np.newaxis, :] - points[np.newaxis, :, :])**2).sum(axis=2))  
    triu_indices = np.triu_indices(n, k=1)  
    close_pairs_indices = np.where(distances[triu_indices] < r)  
  
    start_list = []
    end_list = [] 
    for idx in zip(*close_pairs_indices):  
        i, j = triu_indices[0][idx], triu_indices[1][idx]  
        start_list.append(i)
        end_list.append(j)
    
    assert len(start_list) == len(end_list)
    return start_list, end_list
        
def show_pts(pts):
    channel = pts.shape[1]
    if channel == 3:
        import matplotlib.pyplot as plt  
        from mpl_toolkits.mplot3d import Axes3D  
        
        fig = plt.figure()  
        ax = fig.add_subplot(111, projection='3d')  
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='b', marker='o')  

        ax.set_xlabel('X Label')  
        ax.set_ylabel('Y Label')  
        ax.set_zlabel('Z Label')  
        ax.set_title('3D Point Cloud')  
         
        ax.grid(False)  
        
        plt.savefig('point_cloud.png', dpi=300)
        
        plt.close(fig)

def overlay_image(base_image, overlay_image_path, overlay_size, position):  
    
    overlay_image = Image.open(overlay_image_path)  
    overlay_image = overlay_image.resize(overlay_size)  
    base_image.paste(overlay_image, position, overlay_image)  

def draw_index(base_image, position_on_figure, index):
     
    overlay_image_path = f'/root/lyt/lmp_proc/data_pool/index_pool/{index}.png'  
    overlay_size = (22, 22) 
    
    centroid_position = (int(position_on_figure[0]-overlay_size[0]/2), int(position_on_figure[1]-overlay_size[1]/2))
    
    overlay_image(base_image, overlay_image_path, overlay_size, centroid_position)


def draw(base_image, position_on_figure_list):
    if isinstance(base_image, str):
        base_image = Image.open(base_image)  
    else:
        base_image = base_image
    for idx, position_on_figure in enumerate(position_on_figure_list):
        draw_index(base_image, position_on_figure=position_on_figure, index=idx)
    base_image.save("/root/lyt/test_mark.png")
    base_image = base_image.convert('RGB')
    img_array = np.array(base_image)  
    import matplotlib.pyplot as plt  
    plt.imshow(img_array)  
    plt.axis('off')  
    plt.show()  

def vis_pcd(pts_list):
    import open3d as o3d
    if len(pts_list) == 1:
        pcd_list = []
        for pts in pts_list:
            if pts.shape[-1] == 3:
                pcd = o3d.geometry.PointCloud()  
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(np.ones_like(pts) * [0, 0, 1])  
            else:
                pcd = o3d.geometry.PointCloud()  
                pcd.points = o3d.utility.Vector3dVector(pts[:,:3])
                weight = pts[:,3:].reshape(-1,1)
                assert weight.shape[0] == pts.shape[0]
                blue = np.ones_like(pts) * np.array([[0,0,1]])
                red = np.ones_like(pts) * np.array([[1,0,0]])
                color = blue * (1-weight) + red * (weight)
                pcd.colors = o3d.utility.Vector3dVector(color)
            pcd_list.append(pcd)
        o3d.visualization.draw_geometries(pcd_list)    
    else:
        attention_pts = np.concatenate(pts_list[1:],axis=0)
        bg_pts = pts_list[0]
        attention_pts_tensor = torch.from_numpy(attention_pts).to(torch.float32)
        bg_pts_tensor = torch.from_numpy(pts_list[0]).to(torch.float32)
        dist = torch.cdist(bg_pts_tensor, attention_pts_tensor)
        min_dist = torch.min(dist,dim=-1)[0]
        attention_index = (min_dist < 0.01).numpy().astype(np.bool8)[:,None]
        pcd = o3d.geometry.PointCloud()  
        pcd.points = o3d.utility.Vector3dVector(bg_pts)
        blue = np.ones_like(bg_pts) * [0, 0, 1]
        red = np.ones_like(bg_pts) * [1, 0, 0]
        color = blue * (1- attention_index) + red * (attention_index)
        pcd.colors = o3d.utility.Vector3dVector(color)  
        o3d.visualization.draw_geometries([pcd])    

def save_pcd(pts_list,save_path):
    attention_pts = np.concatenate(pts_list[1:],axis=0)
    bg_pts = pts_list[0]
    attention_pts_tensor = torch.from_numpy(attention_pts).to(torch.float32)
    bg_pts_tensor = torch.from_numpy(pts_list[0]).to(torch.float32)
    dist = torch.cdist(bg_pts_tensor, attention_pts_tensor)
    min_dist = torch.min(dist,dim=-1)[0]
    attention_index = (min_dist < 0.003).numpy().astype(np.bool8)[:,None]
    blue = np.ones_like(bg_pts) * [0, 0, 255]
    red = np.ones_like(bg_pts) * [255, 0, 0]
    color = blue * (1- attention_index) + red * (attention_index)
    np.savetxt(save_path, np.concatenate([bg_pts, color], axis=-1))

def save_sim(sim_pcd,save_path):
    bg_pts = sim_pcd[:,:3]
    attention_index = sim_pcd[:,-1].reshape(-1,1)
    blue = np.ones_like(bg_pts) * [0, 0, 255]
    red = np.ones_like(bg_pts) * [255, 0, 0]
    color = blue * (1- attention_index) + red * (attention_index)
    np.savetxt(save_path, np.concatenate([bg_pts, color], axis=-1))

    

if __name__ == "__main__":




    
    if False:
        iterate_time = 1
        device = "cuda:1"
        idx = 2
        # fusion = Fusion(num_cam=4, feat_backbone="dinov2", dtype=torch.float16, device=device)
        data_path = f"/root/lyt/test_episode_{idx}.hdf5"
        # data_path = "/mnt/data/diffusion_data/collection_data/hang_mug_demo_360_multi/episode_2.hdf5"
        data_dict, _ = load_dict_from_hdf5(data_path)
        next_feat_list = [np.load("/root/lyt/test_feature_mug.npy")]#,np.load("/root/lyt/test_feature_tree.npy")]
        for t in range(150):
            if t % 1 == 0:
                vertice = np.load(f"/root/lyt/test_obs_{idx}.npy")[t,0:1000,:3]
                feat_list = next_feat_list
                attention_pts_list, next_feat_list, sim_pcd = draw_fig(data_dict, vertice, fusion, device, feat_list, t, iterate_time)
                save_path = f"test_process_result/attention_{t}.txt"
                vertice = np.load(f"/root/lyt/test_obs_{idx}.npy")[t,0:4000,:3]
                # save_pcd([vertice]+attention_pts_list, save_path)
                save_sim(sim_pcd, save_path)
                print(f"process :{t}")
            

    # vis_pcd([vertice]+attention_pts_list)

    if True:
        data_path = "/mnt/data/diffusion_data/collection_data/episode_1.hdf5"
        data_dict, _ = load_dict_from_hdf5(data_path)
        cam_name_list = [
            "left_bottom_view",
            "right_bottom_view",
            "left_top_view",
            "right_top_view",
        ]

        
        for t in range(150):
            colors = []
            for i in cam_name_list:
                img = data_dict["observations"]["images"][f"{i}_color"][t]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                colors.append(img)
            colors = np.stack(colors, axis=0)

            base_image = Image.fromarray(colors[1], 'RGB')
            base_image.save(f"test_process/test_{t}.png")  
            print(f"test_process/test_{t}.png")

    
    
