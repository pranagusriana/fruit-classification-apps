import numpy as np
from skimage.filters import threshold_otsu
import skimage
import cv2
import os

mapping_idx = {0: 'Apel', 1: 'Jeruk', 2: 'Melon', 3: 'Pear', 4: 'Pisang'}

def get_freq(img, gray_level=256):
    freq = [0 for i in range(gray_level)]
    for pixel in img.reshape(-1):
        freq[pixel] += 1
    return np.array(freq)

def histogram_equalization(img, gray_level=256):
    freq = get_freq(img, gray_level=gray_level)
    n = len(img.reshape(-1))
    norm_freq = freq/n
    s = [0 for i in range(len(freq))]
    s[0] = norm_freq[0]
    for i in range(1, len(norm_freq)):
        s[i] = norm_freq[i] + s[i-1]
    s = (np.array(s) * (len(freq)-1)).astype(int)
    new_img = [[0 for j in range(img.shape[1])] for i in range(img.shape[0])]
    i = 0
    for row in img:
        j = 0
        for col in row:
            new_img[i][j] = s[col]
            j += 1
        i += 1
    return np.array(new_img)

def load_image(file_name, img_size=300):
    sample_image = cv2.imread(file_name)
    img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(img_size, img_size))

    return img

def apply_otsu_method(img):
    img_gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = histogram_equalization(img_gray).astype(np.uint8)
    img_gray = skimage.filters.gaussian(img_gray, sigma=5.0)

    thresh = threshold_otsu(img_gray)
    img_otsu  = img_gray > thresh if thresh > 0.4825 and thresh < 0.5175 else img_gray < thresh
    return img_otsu

def filter_image(image, mask):
    r = image[:,:,0] * mask
    g = image[:,:,1] * mask
    b = image[:,:,2] * mask

    return np.dstack([r,g,b])

def preprocess_image_from_path(file_name):
    img = load_image(file_name)
    mask = apply_otsu_method(img)
    preprocessed_img = filter_image(img, mask)
    return img, mask, preprocessed_img

def preprocess_image(img):
    mask = apply_otsu_method(img)
    preprocessed_img = filter_image(img, mask)
    return preprocessed_img

def load_images_from_path(data_path, img_size=300):
    labels = os.listdir(data_path)
    X = []
    y = []
    for label in labels:
        label_path = os.path.join(data_path, label)
        file_names = os.listdir(label_path)
        for file_name in file_names:
            file_path = os.path.join(label_path, file_name)
            X.append(load_image(file_path, img_size=img_size))
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    print(f"Found {len(X)} images belonging to {len(np.unique(y))} classes.")
    return X, y

def preprocess_image_pipeline(images):
    X = []
    for image in images:
        prep_img = preprocess_image(image)
        features = get_freq(prep_img)
        X.append(features)
    return np.array(X)