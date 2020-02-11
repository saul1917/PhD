import sys
import io
from lxml import etree

import torch
from torch.utils.data.dataset import Dataset
import torchvision
import numpy as np
from PIL import Image

from .files import *
from .classhierarchy import *

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        #print(path)
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS) and os.path.getsize(filename) > 0

def build_file_dictionary(root, root_has_subfolders):
    file_list = []
    if not root_has_subfolders:
        for path, _, files in os.walk(root):
            for filename in files:
                if is_image_file(os.path.join(path, filename)):
                    file_list.append(os.path.join(path, filename))
    else:
        for class_name in get_immediate_subdirectories(root):
            class_folder = os.path.join(root, class_name)
            for filename in get_file_list(class_folder):
                if is_image_file(os.path.join(class_folder, filename)):
                    file_list.append(os.path.join(class_folder, filename))
    return file_list

class MultiLabelDataset(Dataset):

    def add_class_on_level(self, class_level_name, class_name):
        return self.hierarchy.add_class_by_name(class_level_name, class_name)

    def add_to_hierarchy(self, attrs):
        self.hierarchy.add_hierarchy_entry(attrs)

    #needs to be implemented for each type of metadara such as XML, json, etc
    def build_file_attrs(self, file_path):
        pass

    def make_dataset(self, metadata_extension, extensions):
        samples = []
        for complete_img_filename in self.file_list:
            if has_file_allowed_extension(complete_img_filename, extensions):
                name, _ = os.path.splitext(complete_img_filename)
                complete_label_filename = os.path.join(self.root, name + metadata_extension)
                attrs = self.build_file_attrs(complete_label_filename)
                if not attrs is None:
                    item = (complete_img_filename, attrs)
                    samples.append(item)
        return samples

    def __init__(self, level_list, root, extensions, hierarchy=None, loader=default_loader, metadata_extension=".xml", transform=None, target_transform=None, root_has_subfolders=True):
        self.root = root
        self.file_list = build_file_dictionary(root, root_has_subfolders)
        #hierarchies, multi level labels
        #self.level_list = level_list
        #self.level_size = len(level_list)
        #self.initialize_hierarchies()
        if hierarchy is None:
            self.hierarchy = ClassHierarchy(level_list)
        else:
            self.hierarchy = hierarchy
        #classes, class_to_idx = find_classes(root)
        samples = self.make_dataset(metadata_extension, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        #samples has path, target, target is a tuple
        self.root = root
        self.loader = loader
        self.extensions = extensions

        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
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

        sample = np.array(sample)
        target = np.array(target, dtype=np.long)
        #print(sample.shape, target.shape)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class XMLMultiLabelDataset(MultiLabelDataset):

    #code for xml
    def build_file_attrs(self, file_path):
        try:
            parser = etree.XMLParser(encoding='utf-8')
            xml_text = open(file_path, "r").read().replace("&", "&amp;")
            doc = etree.parse( io.BytesIO(xml_text.encode()), parser )
            attrs = []
            for level_name in self.hierarchy.class_level_names: #self.level_list:
                elem = doc.find(level_name)
                idx = self.add_class_on_level(level_name, elem.text)
                if elem is None or elem.text is None or idx is None:
                    return None
                attrs.append(idx)
            attrs = np.array(attrs, np.int32)
            self.add_to_hierarchy(attrs)
            return attrs
        except Exception as e:
            print("Cannot read xml file", file_path, e)
            return None

    def __init__(self, level_list, root, hierarchy=None, transform=None, target_transform=None, loader=default_loader, root_has_subfolders=True):
        MultiLabelDataset.__init__(self, level_list, root, IMG_EXTENSIONS, hierarchy, loader,
                                        metadata_extension=".xml",
                                        transform=transform,
                                        target_transform=target_transform,
                                        root_has_subfolders=root_has_subfolders)
        self.imgs = self.samples
        print(self.hierarchy)
