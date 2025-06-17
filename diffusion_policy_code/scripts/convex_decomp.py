import os

vhacd_path = "/home/yixuan/v-hacd/app/build/TestVHACD"
sapien_assets_dir = "/home/yixuan/bdai/general_dp/sapien_env/sapien_env/assets/yx/"

obj_ls = ['aluminum_mug', 'beer_mug', 'black_mug', 'crusader_mug', 'white_mug', 'wood_mug']

for obj in obj_ls:
    cmd = f'{vhacd_path} {sapien_assets_dir}{obj}/{obj}.obj'
    #mv_obj_cmd = f'mv decomp.obj {sapien_assets_dir}{obj}/'
    #mv_mtl_cmd = f'mv decomp.mtl {sapien_assets_dir}{obj}/'
    #mv_stl_cmd = f'mv decomp.stl {sapien_assets_dir}{obj}/'
    os.system(cmd)
    #os.system(mv_obj_cmd)
    #os.system(mv_mtl_cmd)
    #os.system(mv_stl_cmd)
