from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# create the base pre-trained model

def build_pretrained(pretrained_model, optimizer: str, loss: str, metrics: list[str], num_classes: int, summary=False):
  K.clear_session()
  base_model = pretrained_model(weights='imagenet', include_top=False)

  # add a global spatial average pooling layer
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  # let's add a fully-connected layer
  x = Dense(1024, activation='relu')(x)
  # and a logistic layer -- let's say we have 200 classes
  predictions = Dense(num_classes, activation='softmax')(x)

  # this is the model we will train
  model = Model(inputs=base_model.input, outputs=predictions)

  # first: train only the top layers (which were randomly initialized)
  # i.e. freeze all convolutional InceptionV3 layers
  for layer in base_model.layers:
      layer.trainable = False

  # compile the model (should be done *after* setting layers to non-trainable)
  model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  if summary:
    model.summary()

  return model