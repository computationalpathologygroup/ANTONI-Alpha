from PIL import Image
import math

class TilePreprocessor:
    def __init__(self, target_size=892):
        self.target_size = target_size

    def preprocess(self, image):
        """
        Splits image into tiles of target_size x target_size.
        Returns a list of PIL Images.
        """
        w, h = image.size
        n_cols = math.ceil(w / self.target_size)
        n_rows = math.ceil(h / self.target_size)

        tiles = []
        for row in range(n_rows):
            for col in range(n_cols):
                left = col * self.target_size
                upper = row * self.target_size
                right = min(left + self.target_size, w)
                lower = min(upper + self.target_size, h)
                
                box = (left, upper, right, lower)
                tile = image.crop(box)
                tiles.append(tile)
        
        return tiles
