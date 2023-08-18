import keras
from timeseriesPredictor.logger import logging
from timeseriesPredictor.config.configuration import PrepareTimeseriesBaseModelConfig
from pathlib import Path
from box import ConfigBox

class PrepareTimeseriesBaseModel:
    def __init__(self, config: PrepareTimeseriesBaseModelConfig):
        self.config = config
    
    @staticmethod
    def get_encoder_layer_size(autoencoder_path:Path):
        autoencoder = keras.models.load_model(autoencoder_path)
        encoderLayer = autoencoder.get_layer("encoder")              
        return ConfigBox({'input_shape': autoencoder.input_shape[1:],
                         'bottleneck_shape': encoderLayer.output_shape[1:]})
    
    @staticmethod
    def save_model(path:Path, model:keras.Model):
        model.save(path)

    @staticmethod
    def _prepare_full_model(learning_rate, input_shape, reshape):      
        
        # Construct the input layer with no definite frame size.
        inputs = keras.layers.Input(shape= input_shape)

        # We will construct 2 `ConvLSTM2D` layers with batch normalization,
        # followed by `Conv3D` and `Conv2D` layer for the spatiotemporal outputs.
        x = keras.layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            recurrent_activation='tanh',
            activation="relu",
            recurrent_dropout=0
        )(inputs)
        x = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x)
        x = keras.layers.TimeDistributed(keras.layers.Dropout(0.25))(x)
        
        x = keras.layers.ConvLSTM2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
            recurrent_activation='tanh',
            recurrent_dropout=0
        )(x)
        x = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x)
        x = keras.layers.TimeDistributed(keras.layers.Dropout(0.25))(x)        
        
        x = keras.layers.Conv3D(
            filters=16, kernel_size=(3, 3, 3), activation="relu", padding="same"
        )(x)

        x = keras.layers.Conv2D(
            filters=16, kernel_size=(3, 3), activation="relu", padding="same"
        )(x)

        x = keras.layers.Reshape(reshape)(x)
        x = keras.layers.Conv2D(
            filters=1, kernel_size=(3, 3), activation="linear", padding="same"
        )(x)        

        # Next, we will build the complete model and compile it.
        model = keras.models.Model(inputs, x)
        model.compile(
            loss='mean_squared_error', 
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['MSE']
            )        
        
        model.summary()
        return model
    
    def update_model(self):
        
        autoencoder_pathes = [self.config.trained_od_autoencoder_path, self.config.trained_tensor_autoencoder_path]
        model_pathes = [self.config.base_od_timeseries_model_path, self.config.base_tensor_timeseries_model_path]
        
        for autoencoder_path, model_path  in zip(autoencoder_pathes, model_pathes):
            row, col, channel =  self.get_encoder_layer_size(autoencoder_path).bottleneck_shape            
            input_shape = [row, col, channel, self.config.params_time_lag]
            row, col, _ =  self.get_encoder_layer_size(autoencoder_path).input_shape
            reshape = [row, col, channel]            

            self.full_model = self._prepare_full_model(        
            learning_rate = self.config.params_learning_rate,
            input_shape = input_shape,
            reshape = reshape            
            )

            self.save_model(path=model_path , model=self.full_model)
            logging.info(f"Timeseries base model saved at {model_path}!")