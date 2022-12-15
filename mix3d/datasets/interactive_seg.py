import logging
from pathlib import Path
from random import random, sample
from typing import Optional, Tuple, Union
from random import choice

import numpy as np
from mix3d.datasets.semseg import SemanticSegmentationDataset, elastic_distortion, crop, flip_in_center
logger = logging.getLogger(__name__)


class InteractiveSegmentationDataset(SemanticSegmentationDataset):
    """Docstring for SemanticSegmentationDataset. """

    def __init__(
        self,
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/scannet",
        label_db_filepath: Optional[
            str
        ] = "configs/scannet_preprocessing/label_database.yaml",
        # mean std values from scannet
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),
        mode: Optional[str] = "train",
        add_colors: Optional[bool] = True,
        add_normals: Optional[bool] = True,
        add_raw_coordinates: Optional[bool] = False,
        add_instance: Optional[bool] = False,
        num_labels: Optional[int] = -1,
        data_percent: Optional[float] = 1.0,
        ignore_label: Optional[Union[int, Tuple[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        place_around_existing=False,
        max_cut_region=0,
        point_per_cut=100,
        flip_in_center=False,
        noise_rate=0.0,
        resample_points=0.0,
        add_unlabeled_pc=False,
    ):
        super().__init__(data_dir=data_dir, label_db_filepath=label_db_filepath, color_mean_std=color_mean_std,
                         mode=mode, add_colors=add_colors, add_normals=add_normals, add_raw_coordinates=add_raw_coordinates, add_instance=add_instance, num_labels=num_labels,
                         data_percent=data_percent, ignore_label=ignore_label, volume_augmentations_path=volume_augmentations_path, image_augmentations_path=image_augmentations_path,
                         place_around_existing=place_around_existing, max_cut_region=max_cut_region, point_per_cut=point_per_cut,
                         flip_in_center=flip_in_center, noise_rate=noise_rate, resample_points=resample_points, add_unlabeled_pc=add_unlabeled_pc)

        self._data = []
        for database_path in self.data_dir:
            database_path = Path(database_path)
            if not (database_path / f"instance_{mode}_database.yaml").exists():
                print(f"generate {database_path}/instance_{mode}_database.yaml first")
                exit()
            db_file = self._load_yaml(database_path / f"instance_{mode}_database.yaml")
            for instances in db_file:
                self._data.extend(instances)
        if data_percent < 1.0:
            self._data = sample(self._data, int(len(self._data) * data_percent))

    def __getitem__(self, idx: int):
        points = np.load(self.data[idx]["original_file"])
        instance_mask = np.load(self.data[idx]["instance_filepath"]).astype(bool)
        points[instance_mask, 9] = 1
        points[~instance_mask, 9] = 0
        coordinates, color, normals, labels = (
            points[:, :3],
            points[:, 3:6],
            points[:, 6:9],
            points[:, 9:],
        )
        simulated_clicks = sample_pos_neg_clicks(coordinates, instance_mask)

        if not self.add_colors:
            color = np.ones((len(color), 3))

        # volume and image augmentations for train
        if "train" in self.mode:
            coordinates -= coordinates.mean(0)
            coordinates += np.random.uniform(coordinates.min(0), coordinates.max(0)) / 2

            if self.flip_in_center:
                coordinates = flip_in_center(coordinates)

            for i in (0, 1):
                if random() < 0.5:
                    coord_max = np.max(points[:, i])
                    coordinates[:, i] = coord_max - coordinates[:, i]
            if random() < 0.95:
                for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
                    coordinates = elastic_distortion(
                        coordinates, granularity, magnitude
                    )
            aug = self.volume_augmentations(
                points=coordinates, normals=normals, features=color, labels=labels,
            )
            coordinates, color, normals, labels = (
                aug["points"],
                aug["features"],
                aug["normals"],
                aug["labels"],
            )
            pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
            color = np.squeeze(self.image_augmentations(image=pseudo_image)["image"])

            if self.point_per_cut != 0:
                number_of_cuts = int(len(coordinates) / self.point_per_cut)
                for _ in range(number_of_cuts):
                    size_of_cut = np.random.uniform(0.05, self.max_cut_region)
                    # not wall, floor or empty
                    point = choice(coordinates)
                    x_min = point[0] - size_of_cut
                    x_max = x_min + size_of_cut
                    y_min = point[1] - size_of_cut
                    y_max = y_min + size_of_cut
                    z_min = point[2] - size_of_cut
                    z_max = z_min + size_of_cut
                    indexes = crop(
                        coordinates, x_min, y_min, z_min, x_max, y_max, z_max
                    )
                    coordinates, normals, color, simulated_clicks, labels = (
                        coordinates[~indexes],
                        normals[~indexes],
                        color[~indexes],
                        simulated_clicks[~indexes],
                        labels[~indexes],
                    )

        # normalize color information
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])

        # prepare labels and map from 0 to 20(40)
        labels = labels.astype(np.int32)
        if labels.size > 0:
            if not self.add_instance:
                # taking only first column, which is segmentation label, not instance
                labels = labels[:, 0].flatten()

        features = color
        if self.add_normals:
            features = np.hstack((features, normals))
        if self.add_raw_coordinates:
            features = np.hstack((features, coordinates))

        features = np.hstack((features, simulated_clicks))

        return coordinates, features, labels


def sample_pos_neg_clicks(coordinates, instance_mask):
    n_clicks = np.random.randint(0, 10)
    n_pos = 1
    pos_prop = round(random())
    n_pos += int(pos_prop * n_clicks)
    n_neg = 10 - n_pos
    mask = instance_mask.copy()
    pos_mask = np.zeros_like(instance_mask).astype(bool)
    neg_mask = np.zeros_like(instance_mask).astype(bool)
    # sample positive clicks
    for _ in range(n_pos):
        pos_mask |= sample_click(coordinates, mask)
        mask[pos_mask] = False
    # sample negative clicks
    mask = instance_mask.copy()
    mask = ~mask
    mask[pos_mask] = False
    for _ in range(n_neg):
        neg_mask |= sample_click(coordinates, mask)
        mask[neg_mask] = False

    return np.hstack((pos_mask.astype(int).reshape(-1, 1), neg_mask.astype(int).reshape(-1, 1)))


def sample_click(coordinates, instance_mask):
    pos_idx = np.where(instance_mask)[0]
    if len(pos_idx) == 0:
        return np.zeros_like(instance_mask).astype(bool)
    click_id = np.random.choice(pos_idx)
    click_coord = coordinates[click_id]
    man_dist = np.abs(coordinates - click_coord)
    click_box_mask = np.sum(man_dist <= 0.05, axis=1) == 3

    return click_box_mask
