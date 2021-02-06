# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset

from pathlib import Path
from sklearn.model_selection import KFold
import glob

trainFolders = [
     'ARMS',
     'AisazuNihaIrarenai',
     'AkkeraKanjinchou',
     'Akuhamu',
     'AosugiruHaru',
     'AppareKappore',
     'Arisa',
     'BEMADER_P',
     'BakuretsuKungFuGirl',
     'Belmondo',
     'BokuHaSitatakaKun',
     'BurariTessenTorimonocho',
     'ByebyeC-BOY',
     'Count3DeKimeteAgeru',
     'DollGun',
     'Donburakokko',
     'DualJustice',
     'EienNoWith',
     'EvaLady',
     'EverydayOsakanaChan',
     'GOOD_KISS_Ver2',
     'GakuenNoise',
     'GarakutayaManta',
     'GinNoChimera',
     'Hamlet',
     'HanzaiKousyouninMinegishiEitarou',
     'HaruichibanNoFukukoro',
     'HarukaRefrain',
     'HealingPlanet',
     "UchiNoNyan'sDiary",
     'UchuKigekiM774',
     'UltraEleven',
     'UnbalanceTokyo',
     'WarewareHaOniDearu',
     'YamatoNoHane',
     'YasasiiAkuma',
     'YouchienBoueigumi',
     'YoumaKourin',
     'YukiNoFuruMachi',
     'YumeNoKayoiji',
     'YumeiroCooking',
     'TotteokiNoABC',
     'ToutaMairimasu',
     'TouyouKidan',
     'TsubasaNoKioku'
]

def get_segmentation(x):
    return '/data/anime2/manga/' + Path(x).parent.name + '/' + Path(x).name.replace('.jpg', '.cache')

class Manga(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=19,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(512, 1024), 
                 center_crop_test=False,
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 fold=0,
                 std=[0.229, 0.224, 0.225]):

        super(Manga, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)
        
        self.root = root
        self.list_path = list_path
        self.fold_index = fold
        self.num_classes = 2
        self.class_weights = torch.FloatTensor([1, 10]).cuda()

        self.multi_scale = multi_scale
        self.flip = flip
        self.center_crop_test = center_crop_test
        

        self.files = self.read_files()

        if num_samples:
            self.files = self.files[:num_samples]
    
    def read_files(self):
        folds = KFold(n_splits = 5, shuffle = True, random_state=42).split(trainFolders, trainFolders)
        files = []
        for idx, (_, valid_indexes) in enumerate(folds):
            if idx == self.fold_index:
                for x in sorted(glob.glob('/data/anime/**/*.jpg', recursive=True)):
                    if Path(get_segmentation(x)).exists() and Path(x).parent.name in trainFolders:
                        folder = Path(x).parent.name
                        idx = trainFolders.index(folder)
                        file = {
                                "img": x,
                                "name": Path(x).parent.name + '/' + Path(x).stem,
                                "weight": 1,
                                "label": get_segmentation(x)
                            }
                        if 'test' == self.list_path and idx in valid_indexes:
                            files.append(file)
                        elif 'train' == self.list_path and idx not in valid_indexes:
                            files.append(file)
                            
        return files
        
    def convert_label(self, label, inverse=False):
        label[label >= 3] -= 3
        label = label != 0
        return label.astype(np.uint8)

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(item["img"], cv2.IMREAD_COLOR)
        size = image.shape

        label = cv2.imread(item["label"], cv2.IMREAD_GRAYSCALE)

        label = self.convert_label(label)

        if 'test' == self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            return image.copy(), label.copy(), np.array(size), name        
        
        
        image, label = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, 
                                self.center_crop_test)

        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).cuda()
                count = torch.zeros([1,1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]
            preds = F.upsample(preds, (ori_height, ori_width), 
                                   mode='bilinear')
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            fname = os.path.join(sv_path, name[i]+'.png')
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            save_img.save(fname)

        
        
