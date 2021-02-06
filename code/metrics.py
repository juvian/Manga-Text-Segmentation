from collections import defaultdict
from fastai.vision import *
from skimage.measure import label, regionprops 
from skimage.morphology import watershed, square
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from itertools import cycle

class NullRecorder():
    def record(self, *args):
        pass

class MetricsCallback(LearnerCallback):
    _order=-20 # Needs to run before the recorder
    def __init__(self, learn, thresh = 0.5):
        self.init = learn is not None

        if self.init: 
            super().__init__(learn)
        
        self.thresh = thresh
        
        self.last_metrics = dict()

        self.datasets = ["easy", "hard"]
        self.modes = ["normal", "ignore"]
        self.start, self.end, self.step = 0, 1, 0.1
        self.bins = np.arange(self.start, self.end, self.step)

    def on_train_begin(self, **kwargs):
        self.clearMetrics()
        self.calculateMetrics()
        if self.init:
            self.learn.recorder.add_metric_names(self.last_metrics.keys())

    def clearMetrics(self):
        self.metrics = defaultdict(float)

        for dataset in self.datasets:
            self.metrics[dataset] = defaultdict(float)
            for mode in self.modes:
                self.metrics[dataset][mode] = defaultdict(float)        

    def bin(self, value):
        return min(int((value - self.start) / self.step), len(self.bins) - 1)

    def on_batch_end(self, train, last_output, last_target, **kwargs):
        if train: return
        
        assert last_output.shape == last_target.shape, f'output shape is {last_output.shape} while target is {last_target.shape}'

        #testing against an image
        if len(last_output.shape) == 3:
            last_output = last_output.unsqueeze(0)
            last_target = last_target.unsqueeze(0)
        
        pred = last_output if last_output.min() >= 0 else torch.sigmoid(last_output) > self.thresh

        expanded = last_target.clone().detach()
        original = expanded.clone()
        original[last_target >= 3] -= 3

        eroded = expanded.clone()
        eroded[last_target >= 3] = 0

        expanded, original, eroded = expanded.squeeze(1).cpu(), original.squeeze(1).cpu(), eroded.squeeze(1).cpu()
        pred = pred.squeeze(1).cpu()
        
        for image in range(0, last_target.shape[0]):
            text_mask = original[image] != 0

            self.metrics["drd_k sum"] += drd(pred[image].unsqueeze(0).unsqueeze(0), text_mask.unsqueeze(0).unsqueeze(0)).sum().item()
            self.metrics["#subn"] += subn(text_mask.numpy())
            self.metrics["psnr"] += psnr(pred[image], text_mask)
            self.metrics["drd"] += self.metrics["drd_k sum"] / max(self.metrics["#subn"], 1) 
            self.metrics["#images"] += 1

            #label text mask = ground truth
            labels = label(original[image].numpy(), connectivity=2)
            regions = regionprops(labels)

            regionsDict = {region.label: region for region in regions}
            
            for region in regionsDict.values():
                region.area # forzes to calculate, seems to be lazy and give 0 later if not done this
                region.dataset = "hard" if original[image][region.coords[0][0], region.coords[0][1]] == 2 else "easy"

            addIntersectArea(regionsDict, labels.copy(), pred[image], "intersectArea")
            #watershedding to match expanded pixels to a single GT
            expansion = addWatershed(regionsDict, labels, expanded[image] != 0, "expandedArea")
            #watershedding to match prediction pixels to a single GT
            predictionWater = addWatershed(regionsDict, labels, pred[image], "predictionArea")
            expansion[expansion != predictionWater] = 0
            addRegionInfo(regionsDict, expansion, "expandedIntersectArea")

            labelsClone = labels.copy()
            addIntersectArea(regionsDict, labelsClone, eroded[image], "erodedArea")
            addIntersectArea(regionsDict, labelsClone, pred[image], "erodedIntersectArea")
            
            intersections = 0

            for region in list(regionsDict.values()):
                if not hasattr(region, "erodedIntersectArea"):
                    region.erodedIntersectArea = 0 
                    region.intersectArea = 0
                    region.expandedIntersectArea = 0

                #there is an edge case where there can be no eroded area intersection because image was cut in half, ignore. We calculate exact metrics with full resolution later
                if not hasattr(region, "erodedArea"):
                    del regionsDict[region.label]
                    continue        

                assert(region.expandedArea >= region.area)    
                
                erodedCoverage = region.erodedIntersectArea / region.erodedArea
                coverage = region.intersectArea / region.area
                
                self.metrics[region.dataset]["normal"]["#tp expanded"] += region.intersectArea
                self.metrics[region.dataset]["normal"]["#tp eroded"] += region.intersectArea

                self.metrics[region.dataset]["ignore"]["#tp expanded"] += region.expandedIntersectArea
                self.metrics[region.dataset]["ignore"]["#tp eroded"] += region.erodedIntersectArea

                self.metrics[region.dataset]["normal"]["#fn"] += region.area - region.intersectArea
                self.metrics[region.dataset]["ignore"]["#fn"] += region.erodedArea - region.erodedIntersectArea

                self.metrics[region.dataset]["#truth"] += 1

                if region.intersectArea > 0:
                    assert(region.expandedIntersectArea <= region.predictionArea)

                    accuracy = region.intersectArea / region.predictionArea
                    expandedAccuracy = region.expandedIntersectArea / region.predictionArea

                    self.metrics[region.dataset]["#intersections"] += 1
                    intersections += 1

                    self.metrics[region.dataset]["ignore"]["#fp"] += region.predictionArea - region.expandedIntersectArea 
                    self.metrics[region.dataset]["normal"]["#fp"] += region.predictionArea - region.intersectArea                     
                else:
                    accuracy = 0
                    expandedAccuracy = 0

                f1 = coverage * accuracy * 2 / max((coverage + accuracy), 1e-6)
                f1Ignore = erodedCoverage * expandedAccuracy * 2 / max((erodedCoverage + expandedAccuracy), 1e-6)

                for mode, word, metric in zip(cycle(["normal", "ignore"]),["accuracy", "accuracy", "coverage", "coverage", "f1", "f1"], [accuracy, expandedAccuracy, coverage, erodedCoverage, f1, f1Ignore]):
                    self.metrics[region.dataset][mode][f'#{word} {self.bin(metric)}'] += 1 
                    self.metrics[region.dataset][mode][word] += metric

            self.metrics["#truth"] += len(regionsDict)        

            #get prediction connected components that did not intersect with text
            pred[image][predictionWater != 0] = 0
            labels = label(pred[image])
            regions = regionprops(labels)

            for region in regions:
                self.metrics["#fp no intersect"] += region.area

            self.metrics["#pred"] += intersections + len(regions)   
                
        
    def on_epoch_begin(self, **kwargs):
        self.clearMetrics()

    def addHistogramMetrics(self, k, metrics, metric):
        for idx, bin in enumerate(self.bins):
            key = f'#{k} {str(round(bin, 2))}-{str(round(bin + self.step, 2))}'
            metrics[key] += metric[f"#{k} {idx}"]

    def calculateMetrics(self):
        
        self.last_metrics["#truth"] = self.metrics["#truth"]
        self.last_metrics["#pred"] = self.metrics["#pred"]
        self.last_metrics["#fp no intersect"] = self.metrics["#fp no intersect"]
        self.last_metrics["#intersections"] = (self.metrics["easy"]["#intersections"] + self.metrics["hard"]["#intersections"])

        self.last_metrics["#subn"] = self.metrics["#subn"]
        self.last_metrics["drd_k sum"] = self.metrics["drd_k sum"]
        self.last_metrics["drd"] = self.last_metrics["drd_k sum"] / max(self.metrics["#subn"], 1e-6)
        self.last_metrics["average drd"] = self.metrics["drd"] / max(self.metrics["#images"], 1e-6)
        self.last_metrics["psnr"] = self.metrics["psnr"] / max(self.metrics["#images"], 1e-6)

        self.last_metrics["global precision quantity %"] = self.last_metrics["#intersections"] / max(self.metrics["#pred"], 1e-6)
        self.last_metrics["global recall quantity %"] = self.last_metrics["#intersections"] / max(self.metrics["#truth"], 1e-6)

        for dataset in self.datasets:
            for metric in ["#intersections", "#truth"]:
                self.last_metrics[dataset + " " + metric] = self.metrics[dataset][metric]

        for mode in self.modes:
            modeMetrics = defaultdict(float)

            for dataset in self.datasets:
                metrics = defaultdict(float)
                modeMetrics[dataset] = defaultdict(float)
                self.addHistogramMetrics("coverage", metrics, self.metrics[dataset][mode])
                self.addHistogramMetrics("accuracy", metrics, self.metrics[dataset][mode])
                self.addHistogramMetrics("f1", metrics, self.metrics[dataset][mode])

                for metric in ["#fn", "#fp", "#tp expanded", "#tp eroded"]:
                    metrics[metric] = self.metrics[dataset][mode][metric]
                    modeMetrics[metric] += metrics[metric]

                metrics["global recall quantity %"] = self.last_metrics[dataset + " #intersections"] / max(self.metrics[dataset]["#truth"], 1e-6)    
                
                metrics["global quality precision %"] = self.metrics[dataset][mode]["accuracy"] / max(self.last_metrics[dataset + " #intersections"], 1e-6)    
                metrics["global quality recall %"] = self.metrics[dataset][mode]["coverage"] / max(self.last_metrics[dataset + " #intersections"], 1e-6)

                GP = metrics["global precision %"] = self.last_metrics["global precision quantity %"] * metrics["global quality precision %"]
                GR = metrics["global recall %"] = metrics["global recall quantity %"] * metrics["global quality recall %"]

                metrics["global f1 %"] = 2 * GR * GP / max((GP + GR), 1e-6)           

                for key in metrics.keys():
                    modeMetrics[dataset][key] = metrics[key]
            
            
            acc, cov = [sum(self.metrics[dataset][mode][metric] for dataset in self.datasets) for metric in ["accuracy", "coverage"]]
            modeMetrics["global quality precision %"] = acc / max(self.last_metrics["#intersections"], 1e-6)
            modeMetrics["global quality recall %"] = cov / max(self.last_metrics["#intersections"], 1e-6)

            GP = modeMetrics["global precision %"] = self.last_metrics["global precision quantity %"] * modeMetrics["global quality precision %"]
            GR = modeMetrics["global recall %"] = self.last_metrics["global recall quantity %"] * modeMetrics["global quality recall %"]
            
            modeMetrics["global f1 score %"] = 2 * GR * GP / max((GP + GR), 1e-6)
            
            modeMetrics["#fp"] += self.metrics["#fp no intersect"]
            P = modeMetrics["pixel precision %"] = modeMetrics["#tp expanded"] / max(modeMetrics["#tp expanded"] + modeMetrics["#fp"], 1e-6)
            R = modeMetrics["pixel recall %"] = modeMetrics["#tp eroded"] / max(modeMetrics["#tp eroded"] + modeMetrics["#fn"], 1e-6)
            modeMetrics["pixel f1 %"] = 2 * P * R / max(P + R, 1e-6)


            for key in modeMetrics.keys():
                if isinstance(modeMetrics[key], float):
                    self.last_metrics[mode + " " + key] = modeMetrics[key]
                else:
                    for k in modeMetrics[key]:
                        self.last_metrics[mode + " " + key + " " + k] = modeMetrics[key][k]

        for key in self.last_metrics.keys():
            self.last_metrics[key] = self.last_metrics[key] * (100 if "%" in key else 1)
            self.last_metrics[key] = int(self.last_metrics[key]) if "#" in key else round(self.last_metrics[key], 2)                                


    def on_epoch_end(self, last_metrics, **kwargs):
        self.calculateMetrics()       
        if self.init:             
            return add_metrics(last_metrics, self.last_metrics.values())    


    def save(self, path, append=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)  
        mode = "a" if append else "w"
        exists = path.exists()
        with path.open(mode) as f:
            if mode == "w" or not exists:
                f.write(','.join(self.last_metrics.keys()))
            f.write('\n' + ','.join([str(stat) if isinstance(stat, int) else '#na#' if stat is None else f'{stat:.6f}' for stat in self.last_metrics.values()]))
    

    def subplots(self, xlabel, ylabel):
        params = {'axes.labelsize': 16,
                'axes.titlesize': 16}
        plt.rcParams.update(params)
        
        rows = (len(self.datasets) * len(self.modes) + 1) // 2
        fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(14, 12))
        axes = axes.flatten()
        indexes = np.arange(len(self.bins))

        for idx, ax in enumerate(axes):
            if idx % 2 == 0 and ylabel is not None:
                ax.set(ylabel=ylabel)
            if idx // 2 == rows - 1 and xlabel is not None:
                ax.set(xlabel=xlabel)    
            ax.set(ylim = [0, 1], yticks = np.arange(0, 1.1, 0.1), yticklabels = [round(bin, 2) for bin in np.arange(0, 1.1, 0.1)])
            ax.set(xticks=indexes - 0.2, xticklabels=[round(bin, 2) for bin in self.bins])
            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
            ax.margins(x=0)

        return fig, axes.flatten()

    def data(self, data):
        l = 1

        if isinstance(data, list):
            data = pd.concat(data)

        if isinstance(data, pd.DataFrame):
            l = len(data.index)
            data = data.sum().to_dict()
        elif isinstance(data, pd.core.groupby.DataFrameGroupBy):
            index = data.mean()["ignore global f1 score %"].idxmax()
            l = int(data.size()[index])    
            data = data.sum().iloc[index]

        return data, l    
                
    def showHistograms(self, configs, metrics, xlabel="Quality Interval", ylabel="Percentage of connected components"):
        totalBars = len(configs * len(metrics))
        fig, axes = self.subplots(xlabel, ylabel)
        axIdx = 0
        indexes = np.arange(len(self.bins))
        patterns = ['/', '*', '-', '+', 'x', '\\', 'o', 'O', '.'][0:totalBars]

        for dataset in self.datasets:
            for mode in self.modes:
                ax = axes[axIdx]
                i = 0
                for config in configs:
                    data, _ = self.data(config['data'])
                    for metric, label in zip(metrics, config['labels']):
                        vals = []

                        for idx, bin in enumerate(self.bins):
                            key = f'{str(round(bin, 2))}-{str(round(bin + self.step, 2))}'
                            vals.append(data[f"{mode} {dataset} #{metric} {key}"])

                        ax.bar(indexes + i / totalBars, np.array(vals) / sum(vals), label = label , width = 1 / totalBars, hatch = patterns[i % len(patterns)])
                        i += 1

                    ax.set(title = dataset + ' - ' + mode.replace("ignore", "relaxed") + " " + f'({int(sum(vals))} connected components)')
                    
                ax.legend(prop={'size': 16})
                axIdx += 1
        
        fig.tight_layout()
        plt.show()    

def getImageMetrics(pred, truth):
    m = MetricsCallback(None)
    m.on_train_begin()    
    m.on_batch_end(False, pred, truth)
    m.calculateMetrics()
    return m

def getDatasetMetrics(dataset, learn):
    m = MetricsCallback(None)
    m.on_train_begin()
    for idx in range(len(dataset.valid.x.items)):
        pred = learn.predict(dataset.valid.x.get(idx, False))[2]
        m.on_batch_end(False, pred, dataset.valid.y.get(idx, False).px)
    m.calculateMetrics() 
    return m   

#modifies labels!
def addIntersectArea(regionsDict, labels, mask, prop):
    labels[mask == 0] = 0
    #Image(tensor(labels).unsqueeze(0)).show(figsize=(15,15), title=prop)
    addRegionInfo(regionsDict, labels, prop)

def addWatershed(regionsDict, labels, mask, prop):        
    distance = nd.distance_transform_edt(mask)
    water = watershed(-distance, labels, mask=mask, connectivity=square(3))
    addRegionInfo(regionsDict, water, prop)
    return water

def addRegionInfo(regionsDict, labels, prop):
    regions = regionprops(labels)

    for region in regions:
        setattr(regionsDict[region.label], prop, region.area)      

def printStats(data, name = "", stats = None, cuts = [], pm = True):
    try:
        index = data.mean()["ignore global f1 score %"].idxmax()
        mean, std = data.mean().iloc[index], data.std().iloc[index]
    except:
        index = 0
        mean, std = data.mean(), data.std()

    if stats is None:
        stats = ["normal pixel f1 %", "normal global f1 score %", "ignore pixel f1 %", "ignore global f1 score %", "normal global quality precision %", "normal global quality recall %", "ignore global quality precision %", "ignore global quality recall %", "global precision quantity %", "global recall quantity %"]
        cuts = [3, 7]

    output = ""

    if name:
        print(name, index)

    for idx, stat in enumerate(stats):
        if not math.isnan(std[stat]) and pm:
            output += f" & ${'%.2f' % mean[stat]} \\pm {'%.1f' % std[stat]}$"
        else:
            output += f" & ${'%.2f' % mean[stat]}$"
        if idx in cuts or idx + 1 == len(stats):
            print(output + " \\\\")
            output = ""

#drd metric
W = np.hypot(np.arange(-2, 3), np.arange(-2, 3)[:, None])
np.reciprocal(W, where=W.astype(bool), out=W)
W /= W.sum()
W = tensor(W).unsqueeze(0).unsqueeze(0).float()

def drd(im, im_gt):
    m1 = (im == 1) & (im_gt == 0)
    m2 = (im == 0) & (im_gt == 1)
    conv1 = F.conv2d((im_gt == 0).float(), W, bias=None, padding=2, stride=(1, 1))
    conv2 = F.conv2d((im_gt == 1).float(), W, bias=None, padding=2, stride=(1, 1))
    return (conv1 * m1 + conv2 * m2).sum(axis=(1,2,3))

def subn(im_gt):
    height, width = im_gt.shape
    block_size = 8
    
    y = im_gt[0:(height//block_size)*block_size, 0:(width//block_size)*block_size]
    sums = y.reshape(y.shape[0]//block_size, block_size, y.shape[1]//block_size, block_size).sum(axis=(1, 3))
    return np.sum((sums != 0) & (sums != (block_size ** 2)))

#psnr metric
def psnr(im, im_gt):
    mse = (im != im_gt).float().mean()
    return 10 * math.log10(1. / mse) if mse > 0 else 100    