import numpy.random as random

class Rectangle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def intersects(self, rect):
        return (self.x < rect.x + rect.width and self.x + self.width > rect.x and
                self.y < rect.y + rect.height and self.y + self.height > rect.y) 
    
    def area(self):
        return self.width * self.height
    
    def __repr__(self):
        return str((self.x, self.y, self.width, self.height))

class RectangleGenerator:
    @staticmethod
    def generate(width, height, limit):
        rects = []

        for i in range(0, min(limit * 2, 15)):
            x, y = random.randint(0, int(width * 0.93)) , random.randint(0, int(height * 0.9))
            
            if random.random_sample() < 0.8:
                w = random.randint(7, 15)
                h = random.randint(10, 35)
            else:
                w = random.randint(15, 100)
                h = random.randint(10, 50)
            
            r = Rectangle(x, y, min(int(w * width / 100), width), min(int(h * height / 100), height))
            add = True
            
            for rect in rects:
                if rect.intersects(r) and random.random_sample() < 0.5:
                    r = Rectangle(x, y, int(r.width / 2), r.height)
                    if rect.intersects(r) and random.random_sample() < 0.5:
                        r = Rectangle(x, y, r.width, int(r.height / 2))
                if rect.intersects(r):
                    add = False
                    break
            
            if add:
                rects.append(r)
                if len(rects) == limit:
                    break
        return rects    