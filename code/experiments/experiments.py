import numpy as np
import torch 
import random
import os
from pathlib import Path
from sklearn.model_selection import KFold

import sys 
sys.path.append('..')

from dataset import get_segmentation, random_crop, SegItemListCustom
from config import *

def random_seed(seed_value, use_cuda = True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

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

assert(len(trainFolders) == len(set(trainFolders)))   

for x in trainFolders:
    assert(os.path.isdir(MASKS_PATH + '/' + x))

def getDatasetLists(dataset):
    return [dataset.train.x, dataset.train.y, dataset.valid.x, dataset.valid.y]

#gets Kfolded data in order to train the models, 4/5 goes to train and 1/5 to validation in each fold. 
def getDatasets(allData, crop=True, padding = 8, cutInHalf = True):
    folds = KFold(n_splits = 5, shuffle = True, random_state=42).split(trainFolders, trainFolders)

    datasets = []

    for _, valid_indexes in folds:
        dataset = (allData
        .split_by_valid_func(lambda x: trainFolders.index(Path(x).parent.name) in valid_indexes)
        .label_from_func(get_segmentation, classes=['text']))
        
        if crop:
            dataset.train.transform([random_crop(size=(512, 800), randx=(0.0, 1.0), randy=(0.0, 1.0))], tfm_y=True)

        for l in getDatasetLists(dataset):
            l.padding = padding
            l.cutInHalf = cutInHalf
        
        datasets.append(dataset)

    return datasets

#returns single dataset to use by methods that were not trained with manga (icdar2013, total-text, synthetic)
def getDataset(allData):
    dataset = (allData
        .split_by_valid_func(lambda _: True)
        .label_from_func(get_segmentation, classes=['text']))
    for l in getDatasetLists(dataset):
        l.padding = 8
        l.cutInHalf = False
    return dataset

#given prediction and ground truth, returns colorized tensor with true positives as green, false positives as red and false negative as white
def colorizePrediction(prediction, truth):
    prediction, truth = prediction[0], truth[0]
    colorized = torch.zeros(4, prediction.shape[0], prediction.shape[1]).int()
    r, g, b, a = colorized[:]
    
    fn = (truth >= 1) & (truth <= 5) & (truth != 3) & (prediction == 0)
    tp = ((truth >= 1) & (truth <= 5)) & (prediction == 1)
    fp = (truth == 0) & (prediction == 1)
    
    r[fp] = 255
    r[fn] = g[fn] = b[fn] = 255
    g[tp] = 255

    a[:, :] = 128
    a[tp | fn | fp] = 255

    return colorized

#given an image index from a folder (like ARMS, 0) finds which dataset has it in validation and the index inside it
def findImage(datasets, folder, index):
    for dIndex, dataset in enumerate(datasets):
        for idx, item in enumerate(dataset.valid.y.items):
            if "/" + folder + "/" in item and str(index) + ".png" in item:
                return dIndex, idx


def getData():
    random_seed(42)
    return (SegItemListCustom.from_folder(MANGA109_PATH)
       .filter_by_func(lambda x: Path(get_segmentation(x)).exists() and Path(x).parent.name in (trainFolders)))