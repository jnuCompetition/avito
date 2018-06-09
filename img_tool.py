from collections import defaultdict
from scipy.stats import itemfreq
from skimage import feature
from PIL import Image as IMG
import numpy as np
import pandas as pd
import operator
import cv2
import os
from keras.preprocessing import image

images_path = './dataset/sample_avito_images/'
imgs = os.listdir(images_path)
features = pd.DataFrame()
features['image'] = imgs




def perform_color_analysis(img, flag):
    def color_analysis(img):
        # obtain the color palatte of the image
        palatte = defaultdict(int)
        for pixel in img.getdata():
            palatte[pixel] += 1

        # sort the colors present in the image
        sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse=True)
        light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
        for i, x in enumerate(sorted_x[:pixel_limit]):
            if all(xx <= 20 for xx in x[0][:3]):  ## dull : too much darkness
                dark_shade += x[1]
            if all(xx >= 240 for xx in x[0][:3]):  ## bright : too much whiteness
                light_shade += x[1]
            shade_count += x[1]

        light_percent = round((float(light_shade) / shade_count) * 100, 2)
        dark_percent = round((float(dark_shade) / shade_count) * 100, 2)
        return light_percent, dark_percent

    im = IMG.open(img)  # .convert("RGB")

    # cut the images into two halves as complete average may give bias results
    size = im.size
    halves = (size[0] / 2, size[1] / 2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        return None

    light_percent = (light_percent1 + light_percent2) / 2
    dark_percent = (dark_percent1 + dark_percent2) / 2

    if flag == 'black':
        return dark_percent
    elif flag == 'white':
        return light_percent
    elif flag == 'all':
        return light_percent,dark_percent
    else:
        return None

def average_pixel_width(img):
    im = IMG.open(img)
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100

def get_dominant_color(img):
    img = cv2.imread(img)
    arr = np.float32(img)
    pixels = arr.reshape((-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    # quantized = palette[labels.flatten()]
    # quantized = quantized.reshape(img.shape)
    dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
    return dominant_color

def get_average_color(img):
    img = cv2.imread(img)
    average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
    return average_color


def getSize(filename):
    st = os.stat(filename)
    return st.st_size

def getDimensions(filename):
    img_size = IMG.open(filename).size
    return img_size

def get_blurrness_score(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(img, cv2.CV_64F).var()
    return fm

def image_classify(model, pak, img_f):
    img = IMG.open(img_f)
    if img.size != (224, 224):
        img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = pak.preprocess_input(x)
    preds = model.predict(x)
    return pak.decode_predictions(preds, top=3)[0]