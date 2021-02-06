import manga109api
from fastai.vision import ImageImageList
import os
import json
from fastai.core import listify
from fastai.basics import ItemBase
from fastai.vision import Image, ImageImageList
from PIL import Image, ImageDraw
from IPython.display import display

def draw_rectangle(img, x0, y0, x1, y1, annotation_type):
    assert annotation_type in ["body", "face", "frame", "text"]
    color = {"body": "#258039", "face": "#f5be41",
            "frame": "#31a9b8", "text": "#cf3721"}[annotation_type]
    width = 1
    draw = ImageDraw.Draw(img)
    draw.line([x0 - width/2, y0, x1 + width/2, y0], fill=color, width=width)
    draw.line([x1, y0, x1, y1], fill=color, width=width)
    draw.line([x1 + width/2, y1, x0 - width/2, y1], fill=color, width=width)
    draw.line([x0, y1, x0, y0], fill=color, width=width)

class MangaImage(ItemBase):
    def __init__(self, image, info, idx):
        self.image = image
        self.data = idx
        self.info = info

    def __str__(self): return str(self.image)

    def apply_tfms(self, tfms, **kwargs):
        for tfm in tfms:
            tfm(self, **kwargs)

        return self  

    def show(self):
        img = Image.open(self.info['path'])
        for text in self.info['text']:
            draw_rectangle(img, text['xmin'], text['ymin'], text['xmax'], text['ymax'], 'text')

        display(img)   


class Manga109ImageList(ImageImageList):
    @classmethod
    def load(cls, path, **kwargs):
        os.makedirs('cache/manga', exist_ok=True)

        try:
            with open('cache/manga/data.json', 'r') as f:
                data = json.load(f)
        except:
            data = manga109api.Parser(root_dir=str(path))  
            data = {'books': data.books, 'annotations': data.annotations}

            with open('cache/manga/data.json', 'w') as f:
                json.dump(data, f)

        try:
            with open('cache/manga/pages.json', 'r') as f:
                pages = json.load(f)
        except:        
            pages = []

            for book in data['books']:
                for page in data['annotations'][book]['book']['pages']['page']:
                    pagedata = {'text': [], 'path': str(path / 'images' / book / (str(page['@index']).zfill(3) + ".jpg"))}
                    if 'text' in page:
                        for txt in page['text'] if isinstance(page['text'], list) else [page['text']]:
                            pagedata['text'].append({'xmin': txt['@xmin'], 'xmax': txt['@xmax'], 'ymin': txt['@ymin'], 'ymax': txt['@ymax'], 'text': txt['#text']})
                    pages.append(pagedata)

            with open('cache/manga/pages.json', 'w') as f:
                json.dump(pages, f)

        return Manga109ImageList(list(filter(lambda x: len(x['text']) > 0, pages)), **kwargs)        

    def get(self, i):
        info = self.items[i]
        image = self.open(info['path'])
        return MangaImage(image, info, i)

    def show_xyzs(self, xs, ys, zs, logger=False, **kwargs):
        if logger:
            logger.show_xyzs(xs, ys, zs, **kwargs)
        else:
            return super().show_xyzs(xs, ys, zs, **kwargs)                     