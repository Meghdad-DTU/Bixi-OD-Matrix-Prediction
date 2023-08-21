from timeseriesPredictor.logger import logging
from timeseriesPredictor.config.configuration import PrepareCNNAutoencoderBaseModelConfig
from pathlib import Path
import keras

class PrepareCNNAutoencoderBaseModel:
    def __init__(self, config: PrepareCNNAutoencoderBaseModelConfig):
        self.config = config

    @staticmethod
    def save_model(path:Path, model:keras.Model):
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
        encoded = keras.layers.Conv2D(filters=18, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        encoded = keras.layers.Conv2D(filters=3, kernel_size=(3, 3), activation='relu', padding='same')(encoded)        
        encoded = keras.layers.BatchNormalization()(encoded)        
        encoded = keras.layers.MaxPooling2D(pool_size=(2, 2))(encoded)
        encoded = keras.layers.Dropout(0.25)(encoded)     
        bottleneck = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='encoder')(encoded)        
        
        # decoder
        decoded = keras.layers.UpSampling2D(upsampling1)(bottleneck)        
        decoded = keras.layers.Conv2D(filters=18, kernel_size=(3, 3), activation='relu', padding='same')(decoded)
        decoded = keras.layers.Conv2D(filters=18, kernel_size=(3, 3), activation='relu', padding='same')(decoded)
        decoded = keras.layers.BatchNormalization()(decoded)             
        decoded = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(decoded)
        decoded = keras.layers.Dropout(0.25)(decoded)
        decoded = keras.layers.UpSampling2D(upsampling2)(decoded)        

        decoded = keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation='linear', padding='same', name='decoder')(decoded)      
        autoencoder = keras.Model(inputs=inputs, outputs=decoded)           

        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss = 'mean_squared_error' ,
            metrics=['MSE']           
                           )
        
        autoencoder.summary()
        return autoencoder
    
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
            logging.info(f"CNN Autoencoder base model saved at {model_path}!")

    

    