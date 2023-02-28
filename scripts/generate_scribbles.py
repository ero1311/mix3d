import yaml
import numpy as np
import os
from scipy.spatial import cKDTree
from tqdm import tqdm
from os.path import exists, join, basename
from multiprocessing import Pool

modes = ['train', 'validation']
scribbles_path = './data/processed/scannet/scribbles'


def sample_scribble(coordinates, instance_mask, radius=0.025, scribble_percent=0.01):
    scribble_length = int(sum(instance_mask) * scribble_percent)
    instance_pts = coordinates[instance_mask]
    final_mask = np.zeros_like(instance_mask, dtype=bool)
    tree = cKDTree(instance_pts)
    point_ind = np.random.choice(instance_pts.shape[0])
    selected_pts = []
    scribble_idx = []
    for i in range(scribble_length):
        curr_point = instance_pts[point_ind]
        selected_pts.append(curr_point)
        point_ball = tree.query_ball_point(
            curr_point, r=radius, return_sorted=True)
        curr_mask = final_mask[instance_mask].copy()
        scribble_idx.extend(point_ball)
        point_dists = []
        for curr_ind in point_ball:
            selected_pts_npy = np.array(selected_pts).reshape(-1, 3)
            dist = sum(np.linalg.norm(selected_pts_npy
                       - instance_pts[curr_ind], ord=2, axis=1))
            point_dists.append(dist)
        point_dists = np.array(point_dists)
        point_dists = point_dists - point_dists.max()
        point_dists = np.exp(point_dists) / np.sum(np.exp(point_dists))
        point_ind = np.random.choice(point_ball, p=point_dists)
    return np.array(list(set(scribble_idx)), dtype='uint16')


def process_instance(instance):
    points = np.load(instance['original_file'])
    instance_mask = np.load(instance['instance_filepath']).astype(bool)
    for scribble_percent in scribble_percents:
        scribble = sample_scribble(points[:, :3], instance_mask, scribble_percent=(scribble_percent / 100))
        np.save(join(scribbles_path, '{}_{}.npy'.format(basename(instance['instance_filepath']).split('.')[0], int(scribble_percent))), scribble)


scribble_percents = np.arange(6, 11)
if not exists(scribbles_path):
    os.makedirs(scribbles_path)

for mode in modes:
    data = []
    with open('./data/processed/scannet/instance_{}_database.yaml'.format(mode)) as f:
        db_file = yaml.safe_load(f)
    for instances in db_file:
        data.extend(instances)
    with Pool(16) as pool:
        max_ = len(data)
        with tqdm(total=max_) as pbar:
            for _ in pool.imap_unordered(process_instance, data):
                pbar.update()
