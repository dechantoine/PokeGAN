from tensorflow import keras

class Preprocessor(keras.models.Sequential):
    def __init__(self, dict_process):
        super().__init__()
        if dict_process["random_flip"]:
            self.add(keras.layers.RandomFlip(mode="horizontal"))
        if dict_process["random_contrast"]:
            self.add(keras.layers.RandomContrast(dict_process["random_contrast_rate"]))