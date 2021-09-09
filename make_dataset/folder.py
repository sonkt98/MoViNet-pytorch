import json

from PIL import Image

import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import make_dataset
from tqdm import tqdm

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def checkAvailiabel(dir, name):
    if not dir == "DOG":
        label_path = os.path.join("/home/petpeotalk/AIHUB/Training/라벨링데이터_" + dir, name + ".json")
        with open(label_path, 'r') as f:
            json_data = json.load(f)
        if json_data['metadata']['location'] == "실내": return True
        else: return False
    else: return True


def find_custom_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    # classes = []
    # for dir in os.listdir(directory):
    #     dir_path = os.path.join(directory, dir)
    #     for entry in os.scandir(dir_path):
    #         if entry.is_dir():
    #             classes.append(entry.name.split('-')[-2])
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    print(classes)
    my_set = set(classes)  # 집합set으로 변환
    classes = list(my_set)  # list로 변환
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(classes))}
    print(class_to_idx)
    return classes, class_to_idx


def make_custom_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).
    See :class:`DatasetFolder` for details.
    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)
    if class_to_idx is None:
        _, class_to_idx = find_custom_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    instances = []
    available_classes = set()
    for target_class in sorted(os.listdir(directory)):
        class_index = class_to_idx[target_class]
        for images_dir in tqdm(sorted(os.listdir(os.path.join(directory, target_class)))):
            path = os.path.join(directory, target_class, images_dir)
            item = path, class_index
            if len(os.listdir(path))<10: continue
            instances.append(item)
        if target_class not in available_classes:
            available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        raise FileNotFoundError(msg)

    return instances

def make_custom_dataset_test(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).
    See :class:`DatasetFolder` for details.
    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    instances = []
    for images_dir in tqdm(sorted(os.listdir(directory))):
        path = os.path.join(directory, images_dir)
        len_images = len(os.listdir(path))
        if len_images < 10: continue
        for i in range(len_images//10):
            item = path, i*10
            instances.append(item)

    return instances
