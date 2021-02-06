import numpy.random as random
from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode

#http://www.rikai.com/library/kanjitables/kanji_codes.unicode.shtml
class TextGenerator:
    ranges = [(33, 122), (0x3040, 0x309f), (0x30a0, 0x30ff), (0xff60, 0xffb0), (0x4e00, 0x9faf)]
    total = sum([x[1] - x[0] + 1 for x in ranges])
    cache = dict()
    
    @staticmethod
    def isChar(c):
        idx = ord(c)
        return any(map(lambda x: idx >= x[0] and idx <= x[1], TextGenerator.ranges))
    
    @staticmethod
    def char(num):
        total = 0
        for x in TextGenerator.ranges:
            total += x[1] - x[0] + 1
            if total > num:
                return chr(x[1] - (total - num) + 1)
    
    @staticmethod
    def generate(l):
       return "".join([TextGenerator.char(random.randint(0, TextGenerator.total)) for x in range(l)])
    
    @staticmethod
    def text_wrap(text, font, max_width, max_height):
        if font.size in TextGenerator.cache:
            char_width = TextGenerator.cache[font.size]
        else:
            char_width = TextGenerator.cache[font.size] = font.getsize('亮')[0]
        estimate = (max_width // char_width)
        lines = []
        i, j, hei = 0, 0, 0
        # append every word to a line while its width is shorter than image width
        while i < len(text) and estimate > 0:
            i = j
            j = min(len(text), i + estimate)
            width = font.getsize(text[i:j])[0]
            while j < len(text) and width <= max_width:
                width += font.getsize(text[j])[0]
                j += 1
            while width > max_width and j > i:
                j -= 1
                width -= font.getsize(text[j])[0]
            hei += font.getsize(text[i:j])[1]
            if hei > max_height or i == j:
                break     
            if len(text[i:j]): 
                lines.append(text[i:j])  
        return lines


from PIL import ImageFont
import json
import os

class Font:
    def __init__(self, path):
        self.cache = dict()
        self.path = path
        self.initAvailableChars()

    def getFont(self, size):
        if size in self.cache:
            font = self.cache[size]
        else:
            font = self.cache[size] = ImageFont.truetype(str(self.path), size)    
        
        return font 

    def initAvailableChars(self):
        self.chars = []
        key = str(self.path.name)
        filename = 'cache/fonts/'+key+'.json'

        try:
            with open(filename, 'r') as f:
                cache = json.load(f)
        except:
            cache = dict()

        if 'ranges' in cache and str(TextGenerator.ranges) == cache['ranges']:
            self.chars = cache['chars']
        else:    
            font = TTFont(str(self.path))
            glyphset = font.getGlyphSet()
            table = font.getBestCmap()

            for r in TextGenerator.ranges:
                #ugly check to know if char is supported by font
                self.chars += [chr(x) for x in range(r[0], r[1] + 1) if x in table.keys() and (glyphset[table[x]]._glyph.bytecode != b' \x1d' if hasattr(glyphset[table[x]]._glyph, 'bytecode') else glyphset[table[x]]._glyph.numberOfContours > 0)]
             
            cache = {'ranges': str(TextGenerator.ranges), 'chars': self.chars}    

            with open(filename, 'w') as f:
                json.dump(cache, f)

        Fonts.total += len(self.chars)        


    def generateText(self, length):
        return "".join(random.choice(self.chars, length))            
        
class Fonts:
    total = 0
    staticmethod
    def load(fontFolder):
        fonts = []
        os.makedirs('cache/fonts', exist_ok=True)
        for extension in ['ttf', 'otf']:
            fonts += [Font(p) for p in fontFolder.glob("**/*." + extension)]

        return fonts    

    def __init__(self, fonts):
        self.fonts = fonts
        self.updateWeights()
    
    def randomFont(self):
        if len(self.weights) != len(self.fonts):
            self.updateWeights()

        return random.choice(self.fonts, p=self.weights) # let fonts with more chars be more likely

    def updateWeights(self):
        self.weights = []
        self.total = 0

        for font in self.fonts:
            self.total += len(font.chars)

        for font in self.fonts:
            self.weights.append(len(font.chars) / self.total)


