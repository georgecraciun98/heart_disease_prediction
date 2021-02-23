from django.apps import AppConfig


class MlAppConfig(AppConfig):
    name = 'ml_app'
    #load your models and model weights here and these are linked      "MODELS" variable that we mentioned in "settings.py"
    # path = os.path.join(settings.MODELS, "my_model03.h5")
    # loaded_model = load_model(path, {"RandomFourierFeatures": RandomFourierFeatures}, False)
    # #loaded_model.load_weights(path)
    print('module was loaded')
