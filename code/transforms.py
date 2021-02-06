from fastai.vision import pil2tensor

from PIL import ImageFont, ImageDraw, Image
from RectangleGenerator import *
from TextGenerator import TextGenerator, Fonts
import numpy as np
from pathlib import Path
import numpy.random as random
from fastai.vision import imagenet_stats, normalize, vision, image2np
import torch
import cv2


def randint(a, b, *args):
    return random.randint(a, b + 1, *args)

def add_spaces(s):
    if len(s) < 3:
        return s
    idxs = set(random.choice(range(0, len(s)), len(s)//10 + 1, False))
    return "".join(map(lambda t: t[1] if t[0] not in idxs else " ",enumerate(s)))

def textify(danImage, fonts):    
    image = danImage.image

    w, h = image.size
    pil_img_x = image.copy()
    pil_img_y = image.copy()
    draw_x = ImageDraw.Draw(pil_img_x, 'RGBA')
    draw_y = ImageDraw.Draw(pil_img_y, 'RGBA')

    padding = randint(4, 10)

    if random.random_sample() < 0.5:
        x, y = randint(0, w // 6), randint(0, h // 6)
        rects = [Rectangle(x, y, randint(w - 2 * x, w - x), randint(h - 2 * y, h - y))]
    else:    
        rects = danImage.rects = RectangleGenerator.generate(w, h, randint(7, 15))

    for rect in rects:

        if random.random_sample() < 0.8:
            text_size = randint(8, 20) 
        else:
            text_size = randint(20, min(w, h) * 7 // 10)

        expected_to_fit = rect.area() // (text_size ** 2)

        font = fonts.randomFont()
        text = font.generateText(randint(expected_to_fit // 2, expected_to_fit + 1))

        sizedFont = font.getFont(text_size)
        
        lines = TextGenerator.text_wrap(text, sizedFont, rect.width - padding * 2, rect.height - padding * 2)
        #lines = list(map(add_spaces, lines))

        border_color = randint(240,255), randint(240,255), randint(240,255)
        
        if random.random_sample() < 0.8:
            text_color = (randint(0,60), randint(0,60), randint(0,60))
        else:
            if random.random_sample() < 0.5:
                text_color = (randint(0,255), randint(0,255), randint(0,255))        
            else:
                text_color = (randint(200,255), randint(200,255), randint(200,255))
                border_color = (randint(0,10), randint(0,10), randint(0,10))

        rotate = random.random_sample() < 0.3
        box = random.random_sample() < 0.1
        angle = randint(0,255)
        alpha = randint(0,200) if random.random_sample() < 0.1 else 255
        border = random.random_sample() < 0.05

        x, y = rect.x + padding, rect.y + padding
        
        if box:
            mask = im = Image.new('RGBA', (rect.width, rect.height), (255, 255, 255, alpha))

            if random.random_sample() < 0.5: # make circle shape
                bigsize = (im.size[0] * 3, im.size[1] * 3)
                mask = Image.new('L', bigsize, 0)
                draw = ImageDraw.Draw(mask) 
                draw.ellipse((0, 0) + bigsize, fill=255)
                mask = mask.resize(im.size, Image.ANTIALIAS)
                im.putalpha(mask)

            if rotate:
                mask = rotate_image(im, angle)

            pil_img_x.paste(im, (rect.x, rect.y), mask) 
            pil_img_y.paste(im, (rect.x, rect.y), mask)   

        if rotate:
            im = Image.new('RGBA', (rect.width, rect.height), (255, 255, 255, 0))
            draw_rotated_text(im, angle, "\n".join(lines), text_color, font=sizedFont)
            pil_img_x.paste(im, (rect.x, rect.y), im)
        else:
            if border:
                draw_border(draw_x, (x, y), "\n".join(lines), border_color, sizedFont)
            draw_x.multiline_text((x, y), "\n".join(lines), fill=text_color, font=sizedFont)         
            
    
    if random.random_sample() < 0.2:
        pil_img_x = pil_img_x.convert('L').convert('RGB')
        pil_img_y = pil_img_y.convert('L').convert('RGB')
    

    danImage.pil_img_x = pil_img_x
    danImage.pil_img_y = pil_img_y

    danImage.x = pil2tensor(danImage.pil_img_x,np.float32).div_(255)

    return danImage

def tensorize(danImage):
    danImage.x_tensor = pil2tensor(danImage.pil_img_x,np.float32).div_(255)
    danImage.y_tensor = pil2tensor(danImage.pil_img_y,np.float32).div_(255)

    return danImage        

def binarize(danImage):

    gray = cv2.cvtColor(np.array(danImage.pil_img_x), cv2.COLOR_RGB2GRAY)
    blur = gray#cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,13,10)
    danImage.pil_img_x = np.stack([thresh, thresh, thresh]).transpose(1,2,0) 

    gray = cv2.cvtColor(np.array(danImage.pil_img_y), cv2.COLOR_RGB2GRAY)
    blur = gray#cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,13,10)
    danImage.pil_img_y = np.stack([thresh, thresh, thresh]).transpose(1,2,0) 

    return danImage 

def rotate_image(im, angle):
    if angle % 90 == 0:
        # rotate by multiple of 90 deg is easier
        return im.rotate(angle)
    else:
        # rotate an an enlarged mask to minimize jaggies
        bigger_mask = im.resize((im.size[0]*8, im.size[1]*8),
                                  resample=Image.BICUBIC)
        return bigger_mask.rotate(angle).resize(
            im.size, resample=Image.LANCZOS)


def draw_rotated_text(image, angle, text, fill, font):
    # get the size of our image
    width, height = image.size
    max_dim = max(width, height)

    # build a transparency mask large enough to hold the text
    mask = Image.new('L', image.size, 0)

    # add text to mask
    draw = ImageDraw.Draw(mask)

    size = draw.multiline_textsize(text, font)
    draw.multiline_text(((width - size[0]) // 2, (height - size[1]) // 2), text, 255, font, align='center')

    rotated_mask = rotate_image(mask, angle)

    # paste the appropriate color, with the text transparency mask
    color_image = Image.new('RGBA', image.size, fill)
    image.paste(color_image, rotated_mask)


def draw_border(draw, pos, text, border, font):
    x, y = pos

    if random.random_sample() < 0.8:
        for adj in range(1, randint(1, 2)):
            #move right
            draw.multiline_text((x-adj, y), text, font=font, fill=border)
            #move left
            draw.multiline_text((x+adj, y), text, font=font, fill=border)
            #move up
            draw.multiline_text((x, y+adj), text, font=font, fill=border)
            #move down
            draw.multiline_text((x, y-adj), text, font=font, fill=border)
            #diagnal left up
            draw.multiline_text((x-adj, y+adj), text, font=font, fill=border)
            #diagnal right up
            draw.multiline_text((x+adj, y+adj), text, font=font, fill=border)
            #diagnal left down
            draw.multiline_text((x-adj, y-adj), text, font=font, fill=border)
            #diagnal right down
            draw.multiline_text((x+adj, y-adj), text, font=font, fill=border)        

def patchify(img):
    w, h = State.getRandomSize()
    wid, hei = img.image.size

    if w < wid:
        if random.random_sample() < 0.6:
            x, y = randint(wid//4, wid//4*3), randint(hei//4, hei//4*3) 
        else:
            x, y = random.randint(0, wid - w), random.randint(0, hei - h)

        x, y = min(x, wid - w), min(y, hei - h)
        img.image = img.image.crop((x, y, x + w, y + h))
        


def mangacrop(manga):
    w, h = State.getRandomSize()

    target = random.choice(manga.info['text'])

    center = ((target['xmin'] + target['xmax']) * 0.5, (target['ymin'] + target['ymax']) * 0.5)

    xmin, ymin = int(center[0] - w // 2), int(center[1] - h // 2)

    xmax = xmin + w
    ymax = ymin + h

    if xmax > manga.image.size[1]:
        xmin -= (xmax - manga.image.size[1])
        xmax -= (xmax - manga.image.size[1])

    if xmin < 0:
        xmax -= xmin
        xmin -= xmin        

    if ymax > manga.image.size[0]:
        ymin -= (ymax - manga.image.size[0])
        ymax -= (ymax - manga.image.size[0])           
    
    if ymin < 0:
        ymax -= ymin
        ymin -= ymin
    
    #print(target, xmin, ymin, xmax, ymax, manga.image.size)

    manga.image = vision.Image(manga.image.px[:,ymin:ymax, xmin:xmax])
    manga.x_tensor = manga.image.px
    manga.y_tensor = manga.x_tensor.clone()


class State:
    minSize, maxSize = 64, 64
    randSizes = None

    @staticmethod
    def getRandomSize():
        if State.randSizes is None:
            State.randSizes = [State.minSize, State.minSize] if State.minSize == State.maxSize else randint(State.minSize//8, State.maxSize//8, 2) * 8
        return State.randSizes

    @staticmethod
    def resetRandomSize():
        State.randSizes = None        
