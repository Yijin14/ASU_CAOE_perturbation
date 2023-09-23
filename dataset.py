# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 23:23:51 2019

@author: Keshik
"""
import torchvision.datasets.voc as voc
import numpy as np
from PIL import Image, ImageDraw
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

class PascalVOC_Dataset(voc.VOCDetection):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

        Args:
            root (string): Root directory of the VOC Dataset.
            year (string, optional): The dataset year, supports years 2007 to 2012.
            image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
                (default: alphabetic indexing of VOC's 20 classes).
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, required): A function/transform that takes in the
                target and transforms it.
    """
    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None):
        
        super().__init__(
             root, 
             year=year, 
             image_set=image_set, 
             download=download, 
             transform=transform, 
             target_transform=target_transform)
    
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
    
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        # return super().__getitem__(index)
        img = Image.open(self.images[index]).convert("RGB")
        # print(img.size)
        # height, width = img.size
        target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        bbox = [(int(target['bbox']['xmin']),int(target['bbox']['ymin'])),(int(target['bbox']['xmax']),int(target['bbox']['ymax']))]
        size = img.size()[-1]
        mask = Image.new("RGB", (size, size))
        img1 = ImageDraw.Draw(mask)  
        img1.rectangle(bbox, fill = 'white')
        # mask.save('mask.png')
        mask = self.transform(mask)
        target['mask'] = mask
        # print(target)
        # exit()

        return img, target
        
    
    def __len__(self):
        """
        Returns:
            size of the dataset
        """
        return len(self.images)
