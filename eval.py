import argparse
import os
import yaml
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from sklearn.manifold import TSNE


parser = argparse.ArgumentParser(
    prog="Model Evaluation",
    description="Evaluates models trained with train.py"
)

parser.add_argument(
    '--data_path',
    '-d',
    type=str,
    help='Directory that contains model weights and config.',
    default='casia'
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

# Open train config
with open(os.path.join(args.model_dir, 'train_config.yaml'), 'r') as file:
    train_config = Box.from_yaml(file)
    
with open('eval_config.yaml', 'r') as file:
    eval_config = Box.from_yaml(file)

training_id = args.model_dir.split('/')[-1]


dataset_name = args.data_path.split("/")[-1]
final_path = os.path.join(args.output_dir, dataset_name, training_id)

tf.random.set_seed(train_config.random_seed)
np.random.seed(train_config.random_seed)
random.seed(train_config.random_seed)
keras.utils.set_random_seed(train_config.random_seed)

train_path = os.path.join(train_config.output_path, training_id)

dataframe = utils.getDataframe(args.data_path, **eval_config)

test_dataframe = dataframe.loc[dataframe['is_test'] == True]
files_test, labels_test = test_dataframe['name'].to_numpy(), test_dataframe['label'].to_numpy()

images_test = utils.load_images(files_test, train_config.image_shape)

model = models.build_pretrained(EfficientNetB0, num_classes=eval_config.num_classes, **train_config.build_model)
model.load_weights(os.path.join(train_path, train_config.save_model.checkpoint_name + '.keras'))

y_pred = model.predict(images_test).argmax(axis=1)
y_test = labels_test

# Tries to create output directory. If it already exists,
# Wipes and recreates it.
try:
    os.makedirs(final_path)
except:
    shutil.rmtree(final_path)
    os.makedirs(final_path)

classes = np.unique(dataframe['label'])
conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix[:eval_config.output_classes, :eval_config.output_classes],
            annot=True,
            fmt='g',
            xticklabels=classes[:eval_config.output_classes],
            yticklabels=classes[:eval_config.output_classes],
    )
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
    
report = classification_report(y_test, y_pred, zero_division=np.nan)
print(report)

with open(os.path.join(final_path,'report.txt'), "w") as report_file:
    report_file.write(report)

plt.savefig(os.path.join(final_path, 'confusion_matrix.png'))

embeds = utils.extract_embeddings(model, images_test, eval_config.embed_layer, eval_config.gradcam_image_num)

X_embeds = TSNE(**eval_config.tsne).fit_transform(embeds)

plt.figure()
plt.scatter(X_embeds[:, 0], X_embeds[:, 1])

plt.savefig(os.path.join(final_path, 'tsne.png'))

for i in range(eval_config.gradcam_image_num):
    heatmap = utils.make_gradcam_heatmap(images_test[i], model, eval_config.embed_layer)
    image = utils.superimpose_gradcam(files_test[i], heatmap)
    image.save(os.path.join(final_path, f'grad-cam-{i}.png'))