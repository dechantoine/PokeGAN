from tensorflow import keras

class Generator(keras.models.Sequential):
    def __init__(self, dict_gen, latent_space_size):
        super().__init__()
        self.add(keras.layers.Dense(((dict_gen["dense"]["pixels"]**2) * dict_gen["dense"]["channels"]),
                                    input_shape=[latent_space_size]))
        self.add(keras.layers.Reshape([dict_gen["dense"]["pixels"],
                                       dict_gen["dense"]["pixels"],
                                       dict_gen["dense"]["channels"]]))
        if dict_gen["dense"]["batchnorm"]:
            self.add(keras.layers.BatchNormalization())

        for c in dict_gen["conv2dts"]:
            self.add(keras.layers.Conv2DTranspose(dict_gen["conv2dts"][c]["channels"],
                                                  kernel_size=dict_gen["conv2dts"][c]["kernel_size"],
                                                  strides=dict_gen["conv2dts"][c]["strides"],
                                                  padding="SAME", activation=keras.layers.LeakyReLU(0.2)))
            if dict_gen["conv2dts"][c]["batchnorm"]:
                self.add(keras.layers.BatchNormalization())

            if dict_gen["conv2dts"][c]["dropout"]:
                self.add(keras.layers.Dropout(dict_gen["conv2dts"][c]["dropout_rate"]))

        self.add(keras.layers.Conv2DTranspose(1, kernel_size=dict_gen["conv2dt_out"]["kernel_size"],
                                              strides=dict_gen["conv2dt_out"]["strides"],
                                              padding="SAME",
                                              activation="sigmoid"))