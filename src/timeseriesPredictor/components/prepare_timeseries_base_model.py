import keras
from timeseriesPredictor.logger import logging
from timeseriesPredictor.config.configuration import PrepareTimeseriesBaseModelConfig
from pathlib import Path

class PrepareTimeseriesBaseModel:
    def __init__(self, config: PrepareTimeseriesBaseModelConfig):
        self.config = config

    @staticmethod
    def save_model(path:Path, model:keras.Model):
        model.save(path)

    @staticmethod
    def _prepare_full_model(learning_rate, input_shape, frame=True):      
        
        # Construct the input layer with no definite frame size.
        inp = keras.layers.Input(shape= input_shape)

        # We will construct 3 `ConvLSTM2D` layers with batch normalization,
        # followed by a `Conv3D` layer for the spatiotemporal outputs.
        x = keras.layers.ConvLSTM2D(
            filters=64,
            kernel_size=(5, 5),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(inp)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ConvLSTM2D(
            filters=64,
            kernel_size=(1, 1),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
        x = keras.layers.Conv3D(
            filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
        )(x)

        # Next, we will build the complete model and compile it.
        model = keras.models.Model(inp, x)
        model.compile(
            loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
            )        
        
        model.summary()
        return model
    
    def update_model(self):
        input_shapes = [self.config.params_od_size, self.config.params_tensor_size]
        model_pathes = [self.config.base_od_model_path, self.config.base_tensor_model_path]
        
        for input_shape, model_path  in zip(input_shapes, model_pathes):
            input_shape = [self.config.params_time_lag -1] + input_shape            
            self.full_model = self._prepare_full_model(        
            learning_rate = self.config.params_learning_rate,
            input_shape = input_shape
            
            )

            self.save_model(path=model_path , model=self.full_model)
            logging.info(f"Timeseries base model saved at {model_path}!")