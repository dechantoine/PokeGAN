from tensorflow import keras

class Discriminator(keras.models.Sequential):
    def __init__(self, dict_disc, pixels):
        super().__init__()
        self.add(keras.layers.Conv2D(dict_disc["conv2d_in"]["channels"],
                                     kernel_size=dict_disc["conv2d_in"]["kernel_size"],
                                     strides=dict_disc["conv2d_in"]["strides"],
                                     padding="SAME", activation=keras.layers.LeakyReLU(0.2),
                                     input_shape=[pixels, pixels, 1]))

        for c in dict_disc["conv2ds"]:
            self.add(keras.layers.Conv2D(dict_disc["conv2ds"][c]["channels"],
                                         kernel_size=dict_disc["conv2ds"][c]["kernel_size"],
                                         strides=dict_disc["conv2ds"][c]["strides"],
                                         padding="SAME", activation=keras.layers.LeakyReLU(0.2)))
            if dict_disc["conv2ds"][c]["batchnorm"]:
                self.add(keras.layers.BatchNormalization())

            if dict_disc["conv2ds"][c]["dropout"]:
                self.add(keras.layers.Dropout(dict_disc["conv2ds"][c]["dropout_rate"]))

        self.add(keras.layers.Flatten())
        self.add(keras.layers.Dense(1, activation="sigmoid"))