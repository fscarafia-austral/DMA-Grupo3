import numpy as np
import cv2
import os
import random
from sklearn.decomposition import PCA
# import tensorflow as tf
# from tensorflow import keras
import math
import matplotlib.pyplot as plt
from mlxtend.preprocessing import standardize

input_dir2 = '.\\train_'

train_img = []
train_label = []
train_label_num = []
labels_unicos_num = []
labels_unicos_texto = []

cont = 0
cant = 0
labels_unicos_texto.append("ad_")
labels_unicos_num.append(cont)
for filename in os.listdir(input_dir2):
    temp = filename[0:3]
    cant += 1
    input_path = os.path.join(input_dir2, filename)
    if (cant > 1):
        if (filename[0:3] != train_label[-1]):
            cont += 1
            labels_unicos_texto.append(filename[0:3])
            labels_unicos_num.append(cont)
    train_label.append(filename[0:3])
    train_label_num.append(cont)
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    train_img.append(img)

train_img_array = np.array(train_img)
#train_label_num.append(22)

print(train_img_array.shape)

print(train_label)

print(len(train_label))

print(train_label_num)

print(len(train_label_num))

print(labels_unicos_num)

print(labels_unicos_texto)