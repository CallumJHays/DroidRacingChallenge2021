from pathlib import Path
from typing import Union, List, Tuple
from xml.etree import ElementTree

import torch
from torch.utils.data import Dataset

from pathlib import Path
import xml.etree.ElementTree as ET
from skimage.draw import polygon2mask
import imageio
import numpy as np

import torch
from torch.utils.data import Dataset

import time

DEFAULT_LABEL = 1


class LabelMeDataset(Dataset):
    """
    Generic dataset for a set of annotations from LabelMe.
    """

    def __init__(self, img_root, ann_root, classes=None):
        self.img_root = Path(img_root)
        self.ann_root = Path(ann_root)

        self.img_paths = []
        self.ann_paths = []

        im_pths = sorted(self.img_root.rglob("*.jpg"))
        for ip in im_pths:
            # Try to find a matching annotation for each jpg
            rel_path = ip.relative_to(self.img_root)
            ann_path = (self.ann_root / rel_path).with_suffix(".xml")
            if not ann_path.exists():
                print("WARNING: Ignoring {}. No matching annotation.".format(rel_path))
                continue

            # Check that there is at least one polygon tag in the annotation
            root = ET.parse(ann_path).getroot()
            # print("Root len: {}".format(len(root)))
            if len(root) < 5:
                print(
                    "WARNING: Ignoring {}. No polygons in annotation.".format(rel_path)
                )
                continue

            self.img_paths.append(ip)
            self.ann_paths.append(ann_path)

        if classes:
            self.classes = ["__background__"] + classes
        else:
            self.classes = ["__background__"] + list(self._get_classes())
            print("Using found classes", self.classes)

        print("Loaded {} Images".format(len(self)))

    @property
    def num_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        raise NotImplementedError


class LabelMeDatasetInstances(LabelMeDataset):
    """
    Dataset for LabelMe annotations.
    Return data in the format for pytorch MaskRCNN Instance Segmentation
    """

    def __init__(self, img_root, ann_root, classes=None, transforms=None):
        super().__init__(img_root, ann_root, classes=classes)
        self.transforms = transforms

    @staticmethod
    def mask_to_bbox(mask):
        pos = np.where(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

    @staticmethod
    def parse_xml(root, img_shape, classes=None):
        """[summary]

        Args:
            root (xml.etree.ElementTree): The XML from labelme

        Returns:
            dict: labels ready for MaskRCNN/ pytorch
        """

        use_all_classes = True
        if classes is not None:
            use_all_classes = False

        masks = []
        labels = []
        bboxes = []
        areas = []
        for obj in root.iter("object"):
            n = obj.find("name").text.lower()
            deleted = int(obj.find("deleted").text)
            if deleted:
                continue
            if not use_all_classes and n not in classes:
                continue
            if use_all_classes:
                ix = DEFAULT_LABEL
            else:
                ix = classes.index(n)
            labels.append(ix)
            pts = []
            poly = obj.find("polygon")
            for pt in poly.iter("pt"):
                x = int(float(pt.find("x").text))
                y = int(float(pt.find("y").text))
                pts.append((y, x))  # y,x
            mask = polygon2mask(img_shape[:2], np.array(pts))
            if np.all(mask == 0):
                #     print('---')
                #     print('Image:', xml_path)
                #     print("Encountered an empty mask, ignoring")
                continue
            masks.append(mask)
            bbox = LabelMeDatasetInstances.mask_to_bbox(mask)
            bboxes.append(bbox)
            # Cal: shouldn't this be mask.sum() ?
            areas.append((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))
        return (
            np.array(masks),
            np.array(labels),
            np.array(bboxes),
            np.array(areas),
        )

    @staticmethod
    def load_from_xml(classes, xml_path, img_shape):
        root = ET.parse(xml_path).getroot()
        return LabelMeDatasetInstances.parse_xml(root, img_shape, classes=classes)

    @staticmethod
    def prepare_target(masks, labels, bboxes, areas, idx):
        n_objs = len(masks)
        target = {}

        # NB: Inputs to torch.as_tensor should be ndarrays not lists
        # Otherwise, 1000x slowdown https://github.com/pytorch/pytorch/issues/13918
        target["boxes"] = torch.as_tensor(bboxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)
        target["image_id"] = torch.as_tensor([idx])
        target["area"] = torch.as_tensor(areas, dtype=torch.float32)
        target["iscrowd"] = torch.zeros((n_objs,), dtype=torch.int64)

        return target

    def __getitem__(self, idx):
        t0 = time.time()
        img = imageio.imread(self.img_paths[idx])
        t1 = time.time()
        masks, labels, bboxes, areas = self.load_from_xml(
            self.classes, self.ann_paths[idx], img.shape
        )
        # TODO: Handle the case of having no label in image
        # t3 = time.time()
        # TODO: what if no label?
        target = self.prepare_target(masks, labels, bboxes, areas, idx)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        # t4 = time.time()
        # print('---------')
        # print('Fetching ', idx)
        # print('imread', t1-t0)
        # print('masks', t2-t1)
        # print('torch', t3-t2)
        # print('transforms', t4-t3)

        return img, target


class InMemoryLabelMeDataset(LabelMeDatasetInstances):
    def __init__(self, Xs, Ys, classes, transforms=None):

        self.Xs = Xs
        self.Ys = Ys
        self.transforms = transforms
        self.classes = ["__background__"] + classes

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):

        img = self.Xs[idx]
        target = self.Ys[idx]

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class CVATExportedAsLabelme3Dataset(Dataset):
    """
    A dataset for CVAT annotations exported in the `labelme 3.0` format.

    Looks for a `labels` folder in a specified `path` parameter.
    Loads the input image from the <filename> and <folder> tags within each label xml, as well as the target as expected by maskrcnn.
    """

    def __init__(
        self,
        path: Union[str, Path],
        classes: List[str],
        ignore_empty: bool = True,
        warn_missing: bool = True,
    ):
        self.classes = ["__background__"] + classes

        path = Path(path)  # normalize path object

        # load a list of (label_path, img_path)
        self.paths: List[Tuple[Path, Path]] = []

        img_warnings = []

        for x in path.glob("labels/*.xml"):
            xml = ElementTree.parse(x)
            # remove the first part of the path, which should point to `path`
            folder = xml.find("folder").text.split("/", 1)[1]
            filename = xml.find("filename").text

            fp = path / f"{folder}/{filename}"
            if not fp.exists():
                img_warnings.append(f"{folder}/{filename}")

            # Only add entries that have been annotated or otherwise if ignore_empty is False
            elif xml.find("object") or not ignore_empty:
                self.paths.append((x, fp))

        if any(img_warnings):
            print(
                f"WARNING: Couldn't find {len(img_warnings)} source images referenced by labels:"
            )
            for warning in img_warnings:
                print(f"\t\t- {warning}")

        assert len(self.paths) > 0

    def __getitem__(self, idx: int):
        assert isinstance(idx, int)

        label_path, img_path = self.paths[idx]

        # read image as a tensor and re-arrange the dimensions
        img = torch.FloatTensor(imageio.imread(img_path))

        # retain the original shape before we permute it because
        # `LabelMeDatasetInstances.load_from_xml` expects (width, height, ch),
        # but maskrcnn needs it in (ch, width, height) form
        orig_img_shape = tuple(img.shape)
        img = img.permute(2, 0, 1)  # then actually permute it

        # remap the image from 0-255 to 0-1
        # TODO: confirm which is correct, this code or the above comment;
        # max(img) != 255 a lot of the time
        img /= torch.max(img)

        # read labels
        masks, labels, bboxes, areas = LabelMeDatasetInstances.load_from_xml(
            self.classes, label_path, orig_img_shape
        )

        # format like maskrcnn wants
        label = LabelMeDatasetInstances.prepare_target(
            masks, labels, bboxes, areas, idx
        )

        return img, label

    def __len__(self):
        return len(self.paths)

    @property
    def num_classes(self):
        return len(self.classes) + 1


class Cached(Dataset):
    "Caching wrapper around another dataset. Only use if underlying dataset is static during runtime."

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.cache = {}

    def __getitem__(self, idx: int):
        if idx in self.cache:
            return self.cache[idx]

        else:
            res = self.dataset[idx]
            self.cache[idx] = res
            return res

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, attr: str):
        return getattr(self.dataset, attr)
