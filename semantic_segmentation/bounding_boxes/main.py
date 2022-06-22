from semantic_segmentation.Real3DAug.tools.datasets import Waymo, SemanticKITTI
from bounding_boxes import BoundingBox
import yaml
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import argparse

__all__ = {
    'Waymo': [Waymo, "../config/waymo.yaml"],
    'SemanticKITTI': [SemanticKITTI, "../config/semantic-kitti.yaml"],
}


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, required=True,
                        help='specify the config for training - Waymo/SemanticKITTI')
    args = parser.parse_args()
    return args


def load_yaml(filename):
    with open(filename, "r") as stream:
        try:
            file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return file


def main():
    args = parse_args()
    if args.cfg_file != ("Waymo" or "SemanticKITTI"):
        raise NotImplementedError
    else:
        cfg = load_yaml((__all__[args.cfg_file])[1])
        dataset = (__all__[args.cfg_file])[0](cfg)

    bbox = BoundingBox()
    for idx, (pcl, _, path, instance, _) in enumerate(tqdm(dataset)):

        if os.path.exists(path):
            os.remove(path)

        u, indices = np.unique(instance, return_inverse=True)

        write_something = False
        for idx_inst, inst in enumerate(u):
            if inst > 0:

                mask = (instance == inst)
                mask = mask.reshape(-1)
                sub_pcl = pcl[mask]
                label = sub_pcl[0, -1]

                tt = bbox.create_bounding_box(sub_pcl[:, :3], label=label, z="bottom", refiment=True)
                if tt != False:
                    bbox.save_txt(path=path)
                    write_something = True
        if not write_something:
            os.makedirs(Path(path).parent, exist_ok=True)
            f = open(path, "a")
            f.close()


if __name__ == "__main__":
    main()
