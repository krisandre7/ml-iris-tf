import argparse
import os
import yaml
import shutil
import tensorflow as tf
import keras
import utils
import numpy as np
import random
import models
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from box import Box
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    prog="Model Evaluation",
    description="Evaluates models trained with train.py"
)

parser.add_argument(
    '--model_dir',
    '-i',
    type=str,
    help='Directory that contains model weights and config.',
    required=True
)
parser.add_argument(
    '--output_dir',
    '-o',
    help="Output directory where generated graphs and metrics are written",
    type=str,
    default='eval'
)

args = parser.parse_args()

# Open config
with open(os.path.join(args.model_dir, 'config.yaml'), 'r') as file:
    train_config = Box.from_yaml(file)

training_id = args.model_dir.split('/')[-1]
final_path = os.path.join(args.output_dir, training_id)

tf.random.set_seed(train_config.random_seed)
np.random.seed(train_config.random_seed)
random.seed(train_config.random_seed)
keras.utils.set_random_seed(train_config.random_seed)
tf.config.experimental.enable_op_determinism()

dataset_path = os.path.join(train_config.dataset_dir, train_config.dataset_name)
train_path = os.path.join(train_config.output_path, training_id)

dataframe = utils.getDataframe(**train_config)

test_dataframe = dataframe.loc[dataframe['is_test'] == True]
files_test, labels_test = test_dataframe['name'].to_numpy(), test_dataframe['label'].to_numpy()

images_test = utils.load_images(files_test, train_config.image_shape)

num_classes = len(np.unique(dataframe['label']))

model = models.build_pretrained(EfficientNetB0, num_classes=num_classes, **train_config.build_model)
model.load_weights(os.path.join(train_path, train_config.save_model.checkpoint_name + '.keras'))

y_pred = model.predict(images_test).argmax(axis=1)
y_test = labels_test

conf_matrix = confusion_matrix(y_test, y_pred)
TP = conf_matrix.diagonal().sum()
FP = np.abs(np.sum(conf_matrix, axis=1).sum() - TP)
FN = np.abs(np.sum(conf_matrix, axis=0).sum() - TP)
TN = np.abs(np.sum(conf_matrix) - (TP + FP + FN))

cf_matrix = np.array([[TP, FP], [FN, TN]])

group_names = ['True Neg','False Pos','False Neg', 'True Pos']
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
plt_labels = [f"{v1}\n\n{v3}" for v1, v3 in
          zip(group_names,group_percentages)]
plt_labels = np.asarray(plt_labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=plt_labels, fmt='')

# Tries to create output directory. If it already exists,
# Wipes and recreates it.
try:
    os.makedirs(final_path)
except:
    shutil.rmtree(final_path)
    os.makedirs(final_path)

plt.savefig(os.path.join(final_path, 'confusion_matrix.png'))