#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image

def get_cat_to_name(cat_file_name):
    cat_to_name = []
    try:
        with open(cat_file_name, 'r') as f:
            cat_to_name = json.load(f)
    except:
        print("Warning: Unable to load category_names")

    return cat_to_name

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    w = 256
    h = 256
    size = 224
    pil_image = Image.open(image)
    pil_image = pil_image.resize((w, h))
    pil_image = pil_image.crop((w/2-size/2, h/2-size/2, w/2+size/2, h/2+size/2))
    np_image = np.array(pil_image)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# Display an image along with the top k classes
def view_image_classes(image, probs, classes, cat_to_name):
    ''' Function for viewing an image and it's predicted classes.
    '''

    fig, (ax1, ax2) = plt.subplots(figsize=(6,6), nrows=2)
    ax1.imshow(Image.open(image).resize((224,224)))
    ax1.axis('off')
    ax1.set_title(cat_to_name[classes[0]])
    yticks = np.arange(len(probs))
    ax2.barh(yticks, probs[::-1])
    ax2.set_aspect(0.1)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([cat_to_name[x] for x in classes[::-1]])
    ax2.set_title('Probability')
    ax2.set_xlim(0, max(probs)+0.1)

    plt.tight_layout()
