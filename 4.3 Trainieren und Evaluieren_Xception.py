import keras
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras import layers
from keras import optimizers, models
from keras.applications import Xception
import numpy as np
import matplotlib.pyplot as plt
import os
import keras.backend as k
from keras.preprocessing import image
import heapq

df = pd.read_excel(os.path.join(os.getcwd(), 'shuffled_file.xlsx'),
                   dtype={'Filename': str, 'Schadensbeschreibung': str, 'Schadensklasse': str})
img_path = 'Ordner mit allen Bildern welche in der Excel-Datei genannt werden'
save_model_path = os.path.join(os.getcwd(), 'Ordner in dem die gesammelten Daten gespeichert werden')
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255,
                             rotation_range=0,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=False,
                             fill_mode='nearest')
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)
class_weights = {0: 1.125,
                 1: 1.0,
                 2: 2.571,
                 3: 3.6,
                 4: 9.692}
i = 0
dataset_size = 335
valset_size = 40
testset_size = 27
acc_list = []
val_acc_list = []
loss_list = []
val_loss_list = []
test_loss_list = []
test_acc_list = []
num_slices = dataset_size / (valset_size + testset_size)
while i < num_slices:
    k.clear_session()

    if i == 0:
        df_train = df.iloc[(i + 1) * (valset_size + testset_size):dataset_size, :]
        df_train.reset_index(drop=True)
    else:
        if i == num_slices - 1:
            df_train = df.iloc[0:dataset_size - (valset_size + testset_size), :]
            df_train.reset_index(drop=True)
        else:
            df_train = pd.concat([df.iloc[0:i * (valset_size + testset_size), :],
                                  df.iloc[(i + 1) * (valset_size + testset_size):dataset_size, :]], sort=True)
            df_train.reset_index(drop=True)
    df_val = df[i * (valset_size + testset_size):((i + 1) * (valset_size + testset_size)) - testset_size]
    df_test = df[i * (valset_size + testset_size) + valset_size:(i + 1) * (valset_size + testset_size)]

    train_generator = datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=img_path,
        x_col="Überschrift der Spalte mit den Dateinamen",
        y_col="Überschrift der Spalte mit den Schadensklassen",
        classes=['Klasse 1', 'Klasse 2', 'Klasse 3', 'Klasse 4', 'Klasse 5'],
        batch_size=67,
        shuffle=False,
        class_mode="categorical",
        target_size=(299, 299))
    validation_generator = test_datagen.flow_from_dataframe(
        dataframe=df_val,
        directory=img_path,
        x_col="Überschrift der Spalte mit den Dateinamen",
        y_col="Überschrift der Spalte mit den Schadensklassen",
        classes=['Klasse 1', 'Klasse 2', 'Klasse 3', 'Klasse 4', 'Klasse 5'],
        batch_size=10,
        shuffle=False,
        class_mode="categorical",
        target_size=(299, 299))
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df_test,
        directory=img_path,
        x_col="Überschrift der Spalte mit den Dateinamen",
        y_col="Überschrift der Spalte mit den Schadensklassen",
        classes=['Klasse 1', 'Klasse 2', 'Klasse 3', 'Klasse 4', 'Klasse 5'],
        batch_size=9,
        shuffle=False,
        class_mode='categorical',
        target_size=(299, 299))
    inp = layers.Input([299, 299, 3])
    model_1 = Xception(weights='imagenet',
                       include_top=False,
                       input_tensor=inp)

    model_1.trainable = True

    set_trainable = False
    for layer in model_1.layers:
        if layer.name == 'block14_sepconv2':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    x = layers.Flatten()(model_1.output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(5, activation='softmax')(x)

    model = models.Model(inp, out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Nadam(lr=1e-5),
                  metrics=['acc'])
    model.summary()
    print('Round: {}'.format(i + 1))
    callbacks_list = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                        factor=0.1,
                                                        patience=2)]
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size
    print(STEP_SIZE_TRAIN)
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=30,
        class_weight=class_weights,
        callbacks=callbacks_list,
        validation_data=validation_generator,
        validation_steps=STEP_SIZE_VALID,
        verbose=2)
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc_list.append(acc)
    val_acc_list.append(val_acc)
    loss_list.append(loss)
    val_loss_list.append(val_loss)
    model.save(save_model_path + 'Schadensgewichtung_img_{}.h5'.format(i + 1))
    test_loss, test_acc = model.evaluate_generator(test_generator)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    print('loss: ', test_loss, ' test_acc: ', test_acc)

    row = 0
    img_name_list = []
    top_pred_list = []
    pred_acc_list = []
    sec_pred_list = []
    sec_pred_acc_list = []
    true_label_list = []
    bereich_list = []
    beschr_list = []
    while row < testset_size:
        path = os.path.join(img_path, str(df_test.iloc[row, 0]))
        img = image.load_img(path, target_size=(299, 299))
        img_tensor = image.img_to_array(img)
        img_tensor_heatmap = img_tensor.astype('float32')
        img_tensor = np.expand_dims(img_tensor_heatmap, axis=0)
        img_tensor = preprocess_input(img_tensor)
        img_tensor /= 255
        pred = model.predict(img_tensor)
        img_name = df_test.iloc[row, 0]
        top_pred = np.argmax(pred) + 1
        pred_acc = round(max(pred[0]) * 100, 2)
        sec_pred = heapq.nlargest(2, range(len(pred[0])), pred[0].__getitem__)[1] + 1
        sec_pred_acc = np.sort(pred[0])
        sec_pred_acc = round(sec_pred_acc[-2] * 100, 2)
        true_label = df_test.iloc[row, 3]
        bereich = df_test.iloc[row, 2]
        beschr = df_test.iloc[row, 1]
        img_name_list.append(img_name)
        top_pred_list.append(top_pred)
        pred_acc_list.append(pred_acc)
        sec_pred_list.append(sec_pred)
        sec_pred_acc_list.append(sec_pred_acc)
        true_label_list.append(true_label)
        bereich_list.append(bereich)
        beschr_list.append(beschr)
        row += 1
    df_pred = pd.DataFrame(data={"Filename": img_name_list, "Top Prediction": top_pred_list,
                                 'Top Accuracy': pred_acc_list, "Sec Prediction": sec_pred_list,
                                 'Sec Accuracy': sec_pred_acc_list,
                                 'True_Label': true_label_list, 'Bereich': bereich_list, 'Beschreibung': beschr_list})
    df_pred.to_csv(save_model_path + 'Predicitons{}.csv'.format(i + 1), sep=';', index=False)
    del model
    i += 1

acc_score = np.average(acc_list, axis=0)
val_acc_score = np.average(val_acc_list, axis=0)
loss_score = np.average(loss_list, axis=0)
val_loss_score = np.average(val_loss_list, axis=0)
df_acc = pd.DataFrame(data={"acc_score": acc_score, "val_acc_score": val_acc_score,
                            'loss_score': loss_score, 'val_loss_score': val_loss_score})
f = open(save_model_path + 'Evaluation.txt', 'w+')
f.write('test_loss: ' + str(test_loss_list) +
        ' test_acc: ' + str(test_acc_list))
df_acc.to_csv(save_model_path + 'Acc_Loss.csv', sep=';', index=False)

print('acc_score: ', acc_score)
print('val_acc_score: ', val_acc_score)
print('loss_score: ', loss_score)
print('val_loss_score: ', val_loss_score)

epochs = range(len(acc_score))

plt.plot(epochs, acc_score, 'ro', label='Training acc')
plt.plot(epochs, val_acc_score, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(save_model_path + 'acc_Schadensgewichtung_img.png')
plt.show()

plt.plot(epochs, loss_score, 'bo', label='Training loss')
plt.plot(epochs, val_loss_score, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(save_model_path + 'loss_Schadensgewichtung_img.png')
plt.show()
