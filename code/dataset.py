from PIL import Image as Im
from skimage.morphology import skeletonize, binary_erosion, binary_dilation, remove_small_objects
from fastai.vision import *
from fastai.vision.data import SegmentationProcessor
from config import *
from skimage.measure import label, regionprops 
from skimage.draw import polygon

def get_segmentation(x):
    folder = x.parent.name
    return MASKS_PATH + '/' + folder + '/' + x.name.replace('.jpg', '.png')

def divround_up(value, step):
    return (value+step-1)//step*step

def pad_tensor(t, multiple = 8):
    padded = torch.zeros(t.shape[0], divround_up(t.shape[1], multiple), divround_up(t.shape[2], multiple))
    padded[:, 0:t.shape[1], 0:t.shape[2]] = t
    return padded

def unpad_tensor(t, shape):
    return t[:, 0:shape[1], 0:shape[2]]       

def _random_crop(px, size, randx:uniform=0.5, randy:uniform=0.5):
    if isinstance(size, int):
        size = (size, size)
    y, x = int((px.shape[1] - size[1]) * randx), int((px.shape[2] - size[0]) * randy)
    return px[:, y:y+size[1], x:x+size[0]]

def cut_tensor(t, left=True):
    return t[:, :, 0:t.size(2)//2].contiguous() if left else t[:, :, t.size(2)//2:].contiguous()

random_crop = TfmPixel(_random_crop)

mapping = dict()
mapping[(1, 1, 1)] = 1 #black (easy)
mapping[(255, 1, 255)] = 2 #pink (hard)
mapping[(255, 255, 255)] = 0 #white (background)


#replaces part of truth with new ignore categories to account for errors during labeling.
def addIgnore(mask, ignore = True):
    if ignore:
        skeleton = torch.zeros(mask.shape).bool()

        #need to make skeletons by dataset type or 2 close pink/black components will end up with same component in the skeleton
        for val in mask[0].unique():
            if val == 0: continue
        
            sk = tensor(skeletonize((mask[0] == val).numpy())) #get skeleton of text
            skeleton |= sk.bool()
        
        newMask = (mask[0] != 0).numpy() #separate into text/non text
        eroded = newMask.copy()
        dilated = newMask.copy()

        for x in range(0, 3):
            eroded = binary_erosion(eroded)
            dilated = binary_dilation(dilated)
        
        eroded = tensor(np.expand_dims(eroded, 0)) 
        dilated = tensor(np.expand_dims(dilated, 0))

        mask[((dilated != 0) * ((eroded == 0) & (skeleton == 0)))] += len(mapping)
    
    return mask

class CustomImageSegment(ImageSegment):
    @property
    def eroded(self):
        px = self.px.clone()
        px[px >= 3] = 0
        return type(self)(px)

    @property
    def original(self):
        px = self.px.clone()
        px[px >= 3] -=3
        return type(self)(px)

    @property
    def boxed(self):
        mask = self.px[0]
        labels = label(mask != 0, connectivity = 2)
        for region in regionprops(labels):
            sub = mask[region.slice[0], region.slice[1]]
            sub[sub == 0] = 3
        return type(self)(mask.unsqueeze(0).float())    
    
    @property
    def rgb(self):
        rgb = torch.zeros(3, self.shape[1], self.shape[2])
        px = self.px[0]
        r, g, b = rgb
        blacks = (px == 1) 
        pinks = (px == 2) 
        dilated = px == 3
        eroded =  (px == 4) | (px == 5)
        bg = (px == 0)
        
        r[bg] = g[bg] = b[bg] = 255
        r[blacks] = g[blacks] = b[blacks] = 1
        g[pinks] = 1
        r[pinks] = b[pinks] = 255
        g[dilated] = 255
        b[eroded] = 255
        
        return Image(rgb.div(255))
        
#caches image processing that happens on open
class CacheProcessor(SegmentationProcessor):
    def process(self, ds):
        super().process(ds)
        for idx, item in enumerate(ds.items):
            path = Path(str(item).replace(".png", ".cache"))
            if not path.exists():
                im = Im.fromarray(image2np(ds.open(item, False).px).astype(np.uint8))
                im.save(path, format='PNG')

#used to make dataloder have twice the size by cutting images by half
class CutInHalf():
    cutInHalf = True
    padding = 8

    def get(self, i, cutInHalf = None):
        if cutInHalf is None:
            cutInHalf = self.cutInHalf
        x =  super().get(i % len(self.items)).px
        
        if cutInHalf:
            x = cut_tensor(x, i < len(self.items))
            
        return self.reconstruct(x)       

    def __len__(self):
        return max(1, len(self.items) * (2 if self.cutInHalf else 1))

class SegLabelListCustom(CutInHalf, SegmentationLabelList):
    ignore = True
    areaThreshold = 3
    _processor = CacheProcessor
    useCached = True

    def reconstruct(self, t):
        return CustomImageSegment(pad_tensor(t, self.padding))
    
    def analyze_pred(self, pred, thresh:float=0.5): 
        return torch.sigmoid(pred) > thresh

    def open(self, fn, useCached=None):
        if useCached is None:
            useCached = self.useCached

        if useCached:
            fn = type(fn)(str(fn).replace(".png", ".cache"))

        im = Im.open(fn)
        tensor = pil2tensor(im, np.uint8)
        
        if ".cache" in str(fn):
            return self.reconstruct(tensor)
    	
        palette = im.getpalette()

        assert(tensor.ge(len(mapping)).sum().item() == 0) #assert we only have 3 colors
                
        newTensor = tensor.clone()
        
        for x in range(tensor.max().item() + 1): #replace color index (which can be black = 0 for one image, black = 1 for another) for something standard, defined in mapping
            color = (palette[x * 3], palette[x * 3 + 1], palette[x * 3 + 2])
            assert color in mapping, str(color) + " not in mapping " + fn #assert label only has the colors we think they have (black, pink, white)          
            newTensor[tensor == x] = mapping[color] 

        return self.reconstruct(addIgnore(self.threshold(newTensor), self.ignore))

    #removes small components and fill in small holes
    def threshold(self, mask):
        if self.areaThreshold is not None:
            #fill in small holes
            labels = label(mask[0].cpu() == 0, connectivity=2)
            for region in regionprops(labels):
                if region.area <= self.areaThreshold:
                    values = mask[0, region.slice[0], region.slice[1]]
                    val = values.max()
                    if val == 0:
                        val = mask[0, tensor(binary_dilation(labels == region.label))].max()
                    values[values == 0] = val

            #remove small components
            labeles = label(mask[0].cpu(), connectivity=2)
            remove_small_objects(labeles, self.areaThreshold + 1, in_place=True)
            mask[0][labeles == 0] = 0    

            #make sure it all worked
            labels = label(mask[0].cpu(), connectivity=2, background = 2)
            for region in regionprops(labels):
                if region.area <= self.areaThreshold:
                    assert False, "didn't work"       


        return mask            



    def loadPrediction(self, path):
        mask = open_mask(path)
        mask.px = (mask.px != 0).float()
        mask.px = pad_tensor(mask.px, self.padding)
        return mask       

    def sickzilImage(self, idx):
        p = Path(self.items[idx])
        return self.loadPrediction(Path(SICKZIL_PATH) / (p.parent.name + "_mproj") / "masks" / p.name)
      
    def yu45020Image(self, idx):
        p = Path(self.items[idx])
        return self.loadPrediction(Path(YU45020_PATH) / (p.parent.name) / p.name.replace(p.suffix, ".jpg"))

    def xceptionImage(self, idx):
        p = Path(self.items[idx])
        pred = torch.load(Path(EXPERIMENTS_PATH) / 'model' / 'xception' /'predictions' / p.parent.name / p.name.replace(p.suffix, ".pt"))
        mask = torch.sigmoid(pred) > 0.5
        mask = pad_tensor(mask, self.padding)
        return ImageSegment(mask)

    #get mask from polygon prediction
    def craftImage(self, idx):
        mask = torch.zeros(self.get(idx, False).px.shape)[0].transpose(1, 0)
        path = Path(self.items[idx])
        folder, name = path.parent.name, path.stem
        with open(Path(CRAFT_PATH) / folder / ('res_' + name + '.txt'), 'r') as f:
            polygons = list(map(lambda x: x.split(','), filter(len, f.read().split('\n'))))
        for poly in polygons:
            rr, cc = polygon(np.array(poly[::2]).astype(np.uint32), np.array(poly[1:][::2]).astype(np.uint32), shape = mask.shape)
            mask[rr, cc] = 1
        return self.reconstruct(tensor(mask.transpose(1, 0)).unsqueeze(0))    

    def cachedPrediction(self, datasetIdx, imageIdx, folder = Path(EXPERIMENTS_PATH) / 'model' / 'resnet34'):
        TENSOR_PATH = folder / str(datasetIdx) / 'predictions' / (str(imageIdx) + '.pt')
        return torch.load(TENSOR_PATH)

    
class SegItemListCustom(CutInHalf, SegmentationItemList):
    _label_cls = SegLabelListCustom
    
    def __init__(self, items, *args, **kwargs):
        items = sorted(items)
        super().__init__(items, **kwargs)

    def reconstruct(self, t):
        return Image(pad_tensor(t, self.padding))    
