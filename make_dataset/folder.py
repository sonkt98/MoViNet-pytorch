import json

from PIL import Image

import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import make_dataset
from tqdm import tqdm


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


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


def find_custom_classes_test(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    # classes = []
    # for dir in os.listdir(directory):
    #     dir_path = os.path.join(directory, dir)
    #     for entry in os.scandir(dir_path):
    #         if entry.is_dir():
    #             classes.append(entry.name.split('-')[-2])
    classes = ['bodyscratch', 'bodyshake', 'turn']
    print(classes)
    my_set = set(classes)  # 집합set으로 변환
    classes = list(my_set)  # list로 변환
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(classes))}
    print(class_to_idx)
    return classes, class_to_idx


def make_custom_dataset_test(
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
            instances.append(item)
        if target_class not in available_classes:
            available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        raise FileNotFoundError(msg)

    return instances

class DatasetFolder(VisionDataset):
    """A generic data loader.
    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).
        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.
        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.
        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.
        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::
            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext
        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.
        Args:
            directory(str): Root directory path, corresponding to ``self.root``
        Raises:
            FileNotFoundError: If ``dir`` has no class folders.
        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


