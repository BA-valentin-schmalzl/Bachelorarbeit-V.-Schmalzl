import keras
import pandas as pd
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras import layers, optimizers
from keras import Input
from keras import Model
import matplotlib.pyplot as plt
from keras.regularizers import l1, l2
import os
from keras.utils import plot_model


df = pd.read_csv('Pfad zur Excel-Datei mit den Bildnamen und zugehörigen Klassen.csv', delimiter=';')
print(df)
columns = ["Überschrift der Spalte mit den Schadensklassen"]
datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                             rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = datagen.flow_from_dataframe(
    dataframe=df[:130],
    directory="Ordner mit allen Bildern",
    x_col="Überschrift der Spalte mit den Dateinamen",
    y_col=columns,
    batch_size=25,
    shuffle=True,
    class_mode="other",
    target_size=(300, 300))
validation_generator = test_datagen.flow_from_dataframe(
    dataframe=df[130:160],
    directory="Ordner mit allen Bildern",
    x_col="Überschrift der Spalte mit den Dateinamen",
    y_col=columns,
    batch_size=32,
    shuffle=True,
    class_mode="other",
    target_size=(300, 300))
test_generator = test_datagen.flow_from_dataframe(
    dataframe=df[160:],
    directory="Ordner mit allen Bildern",
    x_col="Überschrift der Spalte mit den Dateinamen",
    batch_size=1,
    shuffle=False,
    class_mode=None,
    target_size=(300, 300))

input_1 = Input(shape=(300, 300, 3), dtype='float32', name='egal')

# This is module with image preprocessing utilities

z = layers.Dense(32, activation='tanh', input_shape=(300, 300, 3))(input_1)
z = layers.MaxPooling2D((3, 3))(z)
a_1 = layers.SeparableConv2D(64, (3, 3), activation='tanh', padding='same')(z)
b_1 = layers.SeparableConv2D(64, (7, 7), activation='tanh', padding='same')(z)
c_1 = layers.SeparableConv2D(64, (5, 5), activation='tanh', padding='same')(z)
z = layers.concatenate([a_1, b_1,  c_1])
z = layers.Conv2D(64, (1, 1))(z)
z = layers.SpatialDropout2D(0.2)(z)
z = layers.MaxPooling2D((3, 3))(z)
a_2 = layers.SeparableConv2D(128, (5, 5), activation='tanh', padding='same')(z)
d_2 = layers.GlobalMaxPooling2D()(a_2)
b_2 = layers.SeparableConv2D(128, (7, 7), activation='tanh', padding='same')(z)
e_2 = layers.GlobalMaxPooling2D()(b_2)
c_2 = layers.SeparableConv2D(128, (9, 9), activation='tanh', padding='same')(z)
f_2 = layers.GlobalMaxPooling2D()(c_2)
z = layers.concatenate([a_2, b_2,  c_2])
x = layers.concatenate([d_2, e_2,  f_2])
z = layers.multiply([z, x])
z = layers.Conv2D(128, (1, 1))(z)
z = layers.MaxPooling2D((3, 3))(z)
a_3 = layers.SeparableConv2D(256, (3, 3), activation='tanh', padding='same')(z)
z = layers.SpatialDropout2D(0.2)(a_3)
z = layers.SeparableConv2D(256, (3, 3), activation='tanh', padding='same')(z)
z = layers.SpatialDropout2D(0.2)(z)
z = layers.SeparableConv2D(256, (3, 3), activation='tanh', padding='same')(z)
z = layers.add([a_3, z])
z = layers.SpatialDropout2D(0.1)(z)
z = layers.BatchNormalization()(z)
z = layers.MaxPooling2D((3, 3))(z)
z = layers.Conv2D(128, (1, 1), padding='same')(z)
z = layers.ReLU()(z)
z = layers.Conv2D(64, (1, 1))(z)
z = layers.LeakyReLU(alpha=0.3)(z)
z = layers.Conv2D(32, (1, 1))(z)
z = layers.ReLU()(z)
z = layers.Flatten()(z)
# Von folgendem Layer werden die Gewichtungen erfasst
z = layers.Dense(32, kernel_regularizer=l1(0.001))(z)
z = layers.ReLU()(z)
model_output_1 = layers.Dense(1, activation='sigmoid')(z)
model = Model(input_1, model_output_1)
model.summary()
model.compile(loss=['binary_crossentropy'],
              optimizer=optimizers.Nadam(lr=1e-2),
              metrics=['acc'])
path = os.path.join(os.getcwd(), 'logs/')
callbacks_list = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    factor=0.2,
                                                    patience=5)]

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=40,
    callbacks=callbacks_list,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VALID,
    verbose=2)
print(len(model.layers))
w = model.layers[37].get_weights()[0]
w = np.concatenate(w, axis=0)
w = np.ndarray.tolist(w)
df = pd.DataFrame(data={'Weights': w})
print(df)
save_model_path = 'Ordner in dem die Gewichtungen als csv-Datei gespeichert werden'
df.to_csv(save_model_path + 'weights.csv', sep=';', index=False, decimal=',')

