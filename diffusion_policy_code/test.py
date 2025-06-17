import h5py
import numpy as np


def color_bar(point):
    np.ones_like(point)
    return 

# check h5py
if True:
    full = list()
    data_dir=f"/home/yitong/diffusion/data_train/data_img_1x1/episode_3.hdf5"     
    with h5py.File(data_dir,'r') as file:
        a = file["observations"]["ee_pos"][:]
        b = file["cartesian_action"][:]
        c = 1
        # [5, 480, 360, 3]

# check rgb
if False:
    rgb_path = "/home/yitong/diffusion/test_rgb.npy"
    import matplotlib.pyplot as plt 
    image_data = np.load(rgb_path)
  
    # image_data = np.transpose(image_data, (1, 2, 0))  
    
    # 显示图像  
    plt.imshow(image_data)  
    plt.axis("on")  # 不显示坐标轴  
    # plt.show()
    plt.savefig("/home/yitong/diffusion/test_rgb.png")

    

if False:
    data=np.loadtxt("/home/sim/general_dp-neo-attention_map/check1.txt")
    data=data.transpose(-1,0)
    np.savetxt("/home/sim/general_dp-neo-attention_map/check1_a.txt",data)


if False:
    i=0
    for j in range(2):
        b= np.load("/home/sim/general_dp-neo-attention_map/eval_image.npy")
        print(b.shape)
        out = b[i][j].transpose(1,0)
        np.savetxt(f"/home/sim/general_dp-neo-attention_map/image_{j}.txt",out)

if False:
    from PIL import Image
    path = "/home/sim/general_dp-neo-attention_map/i.npy"
    data = np.load(path)
    data_new=np.zeros_like(data)
    data_new[:,:,0]=data[:,:,-1]
    data_new[:,:,-1]=data[:,:,0]
    data_new[:,:,1]=data[:,:,1]
    image = Image.fromarray(data, "RGB")
    image.save("a.jpg")


# check mask
if False:
    if True:
        path = "/media/yitong/932e6800-38b1-46b9-a874-381bb69f0e77/diff_llm/mask.npy"
        mask=np.load(path)
        print(mask.shape)
        import matplotlib.pyplot as plt
        from PIL import Image  
        image_path = "/media/yitong/932e6800-38b1-46b9-a874-381bb69f0e77/diff_llm/data/fields/pack/color_3.png"

        image = Image.open(image_path)  
  
        # 将图像转换为RGB（如果它是RGBA，则去除alpha通道）  
        if image.mode == 'RGBA':  
            rgb_image = image.convert('RGB')  
        else:  
            rgb_image = image  
        
        # 将图像转换为NumPy数组  
        image_array = np.array(rgb_image)  
        index = 1
        plt.imshow(image_array*mask[index][...,None])
        plt.show()
        plt.savefig("test_mask.jpg")


