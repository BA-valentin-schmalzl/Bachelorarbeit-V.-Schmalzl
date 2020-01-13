import keras
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras import layers
from keras import Input
from keras import Model
import matplotlib.pyplot as plt

df = pd.read_csv('Pfad zur csv-Datei mit den Bildnamen und zugehörigen Klassen.csv', delimiter=';')
print(df)
columns = ["Überschrift der Spalte mit den Schadensklassen"]
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = datagen.flow_from_dataframe(
    dataframe=df[:550],
    directory="Ordner mit allen Bildern",
    x_col="Überschrift der Spalte mit den Dateinamen",
    y_col=columns,
    batch_size=25,

    shuffle=False,
    class_mode="other",
    target_size=(200, 200))
validation_generator = test_datagen.flow_from_dataframe(
    dataframe=df[550:650],
    directory="Ordner mit allen Bildern",
    x_col="Überschrift der Spalte mit den Dateinamen",
    y_col=columns,
    batch_size=32,

    shuffle=True,
    class_mode="other",
    target_size=(200, 200))
test_generator = test_datagen.flow_from_dataframe(
    dataframe=df[650:],
    directory="Ordner mit allen Bildern",
    x_col="Überschrift der Spalte mit den Dateinamen",
    batch_size=1,

    shuffle=False,
    class_mode=None,
    target_size=(200, 200))

input_1 = Input(shape=(200, 200, 3), dtype='float32', name='egal')

# This is module with image preprocessing utilities

x = layers.SeparableConv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3))(input_1)
x = layers.MaxPooling2D((3, 3))(x)
x = layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
z = layers.SeparableConv2D(64, (3, 3), activation='relu')(x)
z = layers.MaxPooling2D((2, 2))(z)
z = layers.SeparableConv2D(128, (3, 3), activation='relu')(z)
z = layers.BatchNormalization()(z)
z = layers.MaxPooling2D((2, 2))(z)
z = layers.SeparableConv2D(256, (3, 3), activation='relu', padding='same')(z)
z = layers.BatchNormalization()(z)
z = layers.MaxPooling2D((2, 2))(z)
z = layers.SeparableConv2D(512, (3, 3), activation='relu')(z)
z = layers.BatchNormalization()(z)
z = layers.Flatten()(z)
z = layers.Dense(512, activation='relu')(z)
z = layers.Dense(64, activation='relu')(z)
model_output_1 = layers.Dense(1, activation='sigmoid')(z)
model = Model(input_1, model_output_1)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

path = 'In diesem Pfad werden die Korrektklassifizierungsraten für die Validierung und das Training ' \
       'sowie die Werte der Verlustfunktion als csv-Datei gespeichert'
callbacks_list = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    factor=0.2,
                                                    patience=5),
                  keras.callbacks.CSVLogger(path + 'Überanpassung.csv',
                                            separator=';',
                                            append=False)]
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=30,
    callbacks=callbacks_list,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VALID,
    verbose=2)

