from PIL import Image
import numpy as np

def save_uint8_image(arr_uint8, path):
    im = Image.fromarray(arr_uint8)
    im.save(path)

def to_grayscale(pil_image):
    return pil_image.convert('L')
