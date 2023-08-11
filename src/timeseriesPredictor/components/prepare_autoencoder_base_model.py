from timeseriesPredictor.logger import logging
from timeseriesPredictor.config.configuration import PrepareAutoencoderBaseModelConfig
from pathlib import Path
import keras

class PrepareAutoencoderBaseModel:
    def __init__(self, config: PrepareAutoencoderBaseModelConfig):
        self.config = config

    @staticmethod
    def save_model(path: Path, model:keras.Model):
        model.save(path)

    @staticmethod
    def _prepare_full_model(learning_rate, input_shape, tensor=False):

        if tensor:
            upsampling1 = (4,4)
            upsampling2 = (2,2)            
        else:
            upsampling1 = (4,4)
            upsampling2 = (2,2)
        
        inputs = keras.layers.Input(shape= input_shape, name='matrix_array') 
        # encoder
        x = keras.layers.Convolution2D(filters=18, kernel_size=(3, 3), padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Dropout(0.25)(x)          
      
        bottleneck = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='bottleneck')(x)        
        
        # decoder
        x = keras.layers.UpSampling2D(upsampling1)(bottleneck)
        x = keras.layers.Convolution2D(filters=9, kernel_size=(3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)      
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = keras.layers.Dropout(0.25)(x)
        x = keras.layers.UpSampling2D(upsampling2)(x)

        decoder = keras.layers.Convolution2D(filters=1, kernel_size=(3, 3), activation='linear', padding='same')(x)      
        full_model = keras.Model(inputs=inputs, outputs=decoder)           

        full_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss = 'mean_squared_error',
            metrics=['MSE'] 

                           )
        
        full_model.summary()
        return full_model
    
    def update_model(self):
        input_shapes = [self.config.params_od_size, self.config.params_tensor_size]
        model_pathes = [self.config.base_od_model_path, self.config.base_tensor_model_path]
        tensors = [False, True]
        for input_shape, model_path, tensor  in zip(input_shapes, model_pathes, tensors):
            self.full_model = self._prepare_full_model(        
            learning_rate = self.config.params_learning_rate,
            input_shape = input_shape,
            tensor = tensor
            )

            self.save_model(path=model_path , model=self.full_model)
            logging.info(f"Autoencoder base model saved at {model_path}!")

    