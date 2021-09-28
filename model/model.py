from .base_model import BaseModel
from dataloader.dataloader import DataLoader

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class MyCNN(BaseModel):
    '''CNN Model Class'''
    def __init__(self, cfg):
        super().__init__(cfg)

    def load_data(self):
        '''loads and preprocess data'''
        (self.ds_train, self.ds_test), self.info = DataLoader().load_data(self.config.data)
        self._preprocess_data()

    def check_data(self):
        '''prints output of dataset, run after .load_data'''
        for [image, label] in self.ds_train.take(1):
            print(f'batch image shape: {image.shape}')
            print(f'batch label shape: {label.shape}')

    def _normalize_img(self, image, label):
        '''Normalizes images: `uint8` -> `float32`.'''
        return tf.cast(image, tf.float32) / 255., label
    
    def _preprocess_data(self):
        autotune = tf.data.experimental.AUTOTUNE

        # process training dataset
        self.ds_train = self.ds_train.map(
            self._normalize_img, num_parallel_calls=autotune)
        self.ds_train = self.ds_train.cache()
        self.ds_train = self.ds_train.shuffle(self.config.train.buffer_size)
        self.ds_train = self.ds_train.batch(self.config.train.batch_size)
        self.ds_train = self.ds_train.prefetch(autotune)

        # process testing dataset
        self.ds_test = self.ds_test.map(
            self._normalize_img, num_parallel_calls=autotune)
        self.ds_test = self.ds_test.batch(self.config.train.batch_size)
        self.ds_test = self.ds_test.cache()                    
        self.ds_test = self.ds_test.prefetch(autotune)

    def build(self):
        '''create and compile model'''
        self.model = tf.keras.Sequential([
            # tf.keras.layers.Input(self.config.data.image_size),
            # tf.keras.layers.Flatten(),
            tf.keras.layers.Flatten(input_shape=self.config.data.image_size),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])

        print(self.model.summary())

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.train.lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy()
        )

    def train(self):
        '''train the model'''
        history = self.model.fit(
            x=self.ds_train,
            epochs=self.config.train.epochs,
            validation_data=self.ds_test
        )

        return history

    def evaluate(self):
        for image, label in self.ds_test.take(1):
            predictions = self.model.predict(image)

        # plot 16 test images and their prediction
        f = plt.figure(figsize=(12,12))
        for i in range(16):
            ax = f.add_subplot(4,4,i+1)
            ax.imshow(image[i])
            pred = np.argmax(predictions[i])
            conf = np.max(predictions[i])
            ax.set_title(f'pred: {pred} (conf:{conf:.2f}), true: {label[i]}')
            
        plt.show()