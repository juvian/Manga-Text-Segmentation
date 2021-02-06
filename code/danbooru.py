from PIL import Image as pilImage
from fastai.vision import Image, ImageImageList, to_float, SegmentationItemList
from fastai.basics import ItemBase
from typing import * 
import torch
import matplotlib.pyplot as plt
from fastai.core import subplots

class DanbooruImage(ItemBase):
    def __init__(self, image, fileDir, idx):
        self.data = idx
        self.image = image
        self.fileDir = fileDir
        
    def __str__(self): return str(self.image)

    def apply_tfms(self, tfms, **kwargs):
        for tfm in tfms:
            tfm(self, **kwargs)

        return self    
        

class DanbooruImageList(ImageImageList):
    @classmethod
    def from_textInfo(cls, textInfo: dict, maxItems=10, maxArea=0, **kwargs):
        gen = filter(lambda k: textInfo[k] <= maxArea, textInfo.keys())
        items = [x for _, x in zip(range(maxItems), gen)]
        return DanbooruImageList(items, **kwargs)

    @classmethod
    def with_text(top, textInfo: dict, **kwargs):
        return DanbooruImageList(sorted(textInfo.keys(), lambda k: textInfo[k])[0:top], **kwargs)    
    
    def get(self, i):
        fileDir = self.items[i]
        image = self.open(fileDir.replace("/danbooru/", "/danbooru_corrupt/"))
        return DanbooruImage(image, fileDir, i)
    
    def open(self, fileDir):
        return pilImage.open(fileDir).convert('RGB')

    def show_xyzs(self, xs, ys, zs, logger=False, **kwargs):
        if logger:
            logger.show_xyzs(xs, ys, zs, **kwargs)
        else:
            return super().show_xyzs(xs, ys, zs, **kwargs) 
    
class DanbooruSegmentationList(SegmentationItemList):
    @classmethod
    def from_textInfo(cls, textInfo: dict, maxItems=10, maxArea=0, **kwargs):
        gen = filter(lambda k: textInfo[k] <= maxArea, textInfo.keys())
        items = [x for _, x in zip(range(maxItems), gen)]
        return DanbooruSegmentationList(items, **kwargs)   

    def get(self, i):
        fileDir = self.items[i]
        image = self.open(fileDir.replace("/danbooru/", "/danbooru_corrupt/"))
        return DanbooruImage(image, fileDir, i)

    def open(self, fileDir):
        return pilImage.open(fileDir).convert('RGB')