import os
import csv
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import yaml
from box import Box
import numpy as np
import glob
import tensorflow as tf

# Rapid prototyping requires flexible data structures, such as dictionaries. 
# However, in Python that means typing a lot of square brackets and quotes. 
# The following trick defines an attribute dictionary that allows us to address keys 
# as if they were attributes:

def openConfig():
    with open('config.yaml', 'r') as file:
        args = yaml.safe_load(file)
    return Box(args)

def load_images(file_paths, input_shape):
    images = np.empty((len(file_paths),) + tuple(input_shape))
    for i in range(len(file_paths)):
      imagem = tf.keras.preprocessing.image.load_img(file_paths[i], target_size=input_shape[:2])
      imagem = tf.keras.preprocessing.image.img_to_array(imagem)
      images[i] = imagem
    return tf.convert_to_tensor(images)

def get_label(path: str, dataset_name: str):
    dataset_split = path.split(dataset_name,1)[1]
    label = int(dataset_split.split('/',2)[1])
    return label - 1

def getDataframe(dataset_dir: str, dataset_name: str, csv_dir: str, train_size: float, test_size: float, random_seed: int, **kwargs):
    csv_name = 'train_test.csv'
    
    dataset = os.path.join(csv_dir, csv_name)
    if os.path.isfile(dataset):
        return pd.read_csv(dataset, index_col=0)

    file_paths = np.array(glob.glob(dataset_dir + dataset_name + '/*/*.bmp'))
    labels = np.array([get_label(path, dataset_name) for path in file_paths])
    files_train, labels_train, files_test, labels_test = stratifiedSortedSplit(file_paths, labels, train_size, test_size, random_seed)
    
    files = np.concatenate([files_train, files_test])
    labels = np.concatenate([labels_train, labels_test])
    dataframe = pd.DataFrame({
        "name": files,
        "label": labels,
    })
    
    dataframe["is_test"] = np.concatenate([np.full(len(files_train), False), np.full(len(files_test), True)])    
    
    dataframe.to_csv(csv_name)
    
    return dataframe

def stratifiedSortedSplit(file_paths: np.array, labels: np.array,
                    train_size: float, test_size: float, random_state: int):
    """Splits image paths and labels equally for each class, then sorts them"""
    splitter = StratifiedShuffleSplit(n_splits=1,
                                      train_size=train_size, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(splitter.split(file_paths, labels))

    files_train, labels_train = file_paths[train_indices], labels[train_indices]
    files_test, labels_test = file_paths[test_indices], labels[test_indices]

    sort_index = np.argsort(labels_train)
    labels_train = labels_train[sort_index]
    files_train = files_train[sort_index]

    sort_index = np.argsort(labels_test)
    labels_test = labels_test[sort_index]
    files_test = files_test[sort_index]

    return files_train, labels_train, files_test, labels_test

def plot_history(history):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(acc))

  plt.figure(figsize=(20,4))
  plt.subplot(1,2,1)
  plt.plot(epochs, acc, label='Training accuracy')
  plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
  plt.title('Training and validation acc')
  plt.legend()
  plt.show()

# Função para carregar as imagens e os rótulos
def carregar_imagens(file_paths, labels, input_shape):
    images = np.empty((len(file_paths),) + input_shape)
    for i in range(len(file_paths)):
      imagem = tf.keras.preprocessing.image.load_img(file_paths[i], target_size=input_shape[:2])
      imagem = tf.keras.preprocessing.image.img_to_array(imagem)
      images[i] = imagem
    return tf.convert_to_tensor(images), tf.convert_to_tensor(labels)