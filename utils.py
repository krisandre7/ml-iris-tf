import os
import csv
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import yaml
from box import Box
import numpy as np
import glob
import tensorflow as tf
import matplotlib.cm as cm

# Rapid prototyping requires flexible data structures, such as dictionaries. 
# However, in Python that means typing a lot of square brackets and quotes. 
# The following trick defines an attribute dictionary that allows us to address keys 
# as if they were attributes:

def openConfig():
    with open('train_config.yaml', 'r') as file:
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

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def superimpose_gradcam(img_path, heatmap, alpha=0.4):

    # Load the original image
    img = tf.keras.utils.load_img(img_path)
    img = tf.keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    return superimposed_img