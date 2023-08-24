import os
import json
import imageio.v2 as imageio
import numpy as np
import cv2
import torch

import torch
import numpy as np

# Translation matrix in the z direction
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

# Rotation matrix around the x-axis (phi is in radians)
rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

# Rotation matrix around the y-axis (theta is in radians)
rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

# Generate a spherical pose transformation matrix
def pose_spherical(theta, phi, radius):
    # Apply translation in the z direction
    c2w = trans_t(radius)
    # Apply rotation around the x-axis (phi)
    c2w = rot_phi(phi) @ c2w
    # Apply rotation around the y-axis (theta)
    c2w = rot_theta(theta) @ c2w
    # Apply an additional transformation (flip) to the resulting matrix
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w




def load_blender(basedir, half_res=False, testskip=1):
    '''
    Load and process Blender dataset images and metadata.

    Parameters:
    basedir (str): The base directory containing dataset files.
    half_res (bool): Option to load half-resolution images if True. Default is False.
    testskip (int): Skipping factor for test data. Default is 1.

    Returns:
    tuple: A tuple containing the processed dataset and related information.
        - imgs (np.ndarray): Numpy array of images in the dataset.
        - poses (np.ndarray): Numpy array of pose transformation matrices relating to each images.
        - render_poses (torch.Tensor): Tensor of pose transformation matrices for rendering (360 degree around the object) - TESTING.
        - image_info (list): List containing [height, width, focal length].
        - index_split (list): List of indices indicating data split boundaries.
    '''
    # List of data splits
    splits = ['train', 'val', 'test']
    # Dictionary to store metadata for each split
    metas = {}
    
    # Load metadata for each split from JSON files
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
    all_imgs = []
    all_poses = []
    counts = [0]
    
    # Loop through each split
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        
        # Determine skipping factor for test data
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip
        
        # Loop through each frame in the metadata
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        
        # Convert images and poses to numpy arrays
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        
        # Record the count of images after each split -> [0, 100, 200, 400]
        counts.append(counts[-1] + imgs.shape[0])        
        
        # Append images and poses to lists
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    # Create index split based on counts
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    # Concatenate all images and poses
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    # # Extract image dimensions and camera parameters
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x']) # AFOV
    focal = .5 * W / np.tan(.5 * camera_angle_x) # as tan(AFOV/2)= (W/2) / Focal length
    
    # # Generate render poses for different angles
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    # Apply half resolution option
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        
    # Return processed data
    return imgs, poses, render_poses, [H, W, focal], i_split

if __name__ == "__main__":
    expname = 'blender_paper_lego'
    basedir = './logs'
    datadir = './data/nerf_synthetic/lego'
    dataset_type = 'blender'

    no_batching = True

    use_viewdirs = True
    white_bkgd = True
    lrate_decay = 500

    N_samples = 64
    N_importance = 128
    N_rand = 1024

    precrop_iters = 500
    precrop_frac = 0.5

    half_res = True
    load_blender(datadir,half_res)