import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import keras
import datetime, os
import yaml
import wandb
import models

import utils
from tensorflow.keras.applications.efficientnet import EfficientNetB0

config = utils.openConfig()

tf.random.set_seed(config.random_seed)
np.random.seed(config.random_seed)
random.seed(config.random_seed)
keras.utils.set_random_seed(config.random_seed)
tf.config.experimental.enable_op_determinism()

dataset_path = os.path.join(config.dataset_dir, config.dataset_name)
training_id =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
final_path = os.path.join(config.output_path, training_id)

dataframe = utils.getDataframe(**config)

test_dataframe = dataframe.loc[dataframe['is_test'] == True]
train_dataframe = dataframe.loc[dataframe['is_test'] == False]
files_test, labels_test = test_dataframe['name'].to_numpy(), test_dataframe['label'].to_numpy()
files_train, labels_train = train_dataframe['name'].to_numpy(), train_dataframe['label'].to_numpy()

images_train = utils.load_images(files_train, config.image_shape)
images_test = utils.load_images(files_test, config.image_shape)

num_classes = len(np.unique(dataframe['label']))
train_count = np.unique(labels_train, return_counts=True)[1].mean()
test_count = np.unique(labels_test, return_counts=True)[1].mean()
print(
    f'Split {train_count} images from each class for train and {test_count} for test')

os.makedirs(final_path, exist_ok=True)

# Write config to output folder
config.to_yaml(os.path.join(final_path, 'config.yaml'))
    
model = models.build_pretrained(EfficientNetB0, num_classes=num_classes, **config.build_model)

tensorboard_callback = tf.keras.callbacks.TensorBoard(final_path, **config.tensorboard)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(final_path, config.save_model.checkpoint_name+ '.keras'),
    **config.save_model)

wandb.tensorboard.patch(root_logdir=config.output_path)
wandb.init(project=config.project_name, config=config, dir=final_path, 
           sync_tensorboard=True)

history = model.fit(
    images_train,
    labels_train,
    validation_data=(images_test, labels_test),
    batch_size=config.batch_size,
    epochs=config.epochs,
    callbacks=[checkpoint, tensorboard_callback]
)

wandb.finish()