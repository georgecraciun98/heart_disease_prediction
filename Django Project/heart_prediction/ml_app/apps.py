import os
import keras
import tensorflow
from django.apps import AppConfig
from django.conf import settings
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from tensorflow.python.keras.layers.kernelized import RandomFourierFeatures

class MlAppConfig(AppConfig):
    name = 'ml_app'
    #load your models and model weights here and these are linked      "MODELS" variable that we mentioned in "settings.py"
    # path = os.path.join(settings.MODELS, "my_model03.h5")
    # loaded_model = load_model(path, {"RandomFourierFeatures": RandomFourierFeatures}, False)
    # #loaded_model.load_weights(path)
    print('module was loaded')
