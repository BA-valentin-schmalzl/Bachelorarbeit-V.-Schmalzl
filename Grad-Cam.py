import pandas as pd
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.applications.imagenet_utils import preprocess_input
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as K
import cv2

save_model_path = os.path.join(os.getcwd(), 'Überordner')
model = load_model(save_model_path + 'Gewichtungen des Modells.h5')
df_test = pd.read_excel(save_model_path + 'Dateipfad zur erstellten Excel-Datei mit den Auswertungen.xlsx')
img_path = 'Dateipfad'
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1. / 255)
row = 108
while row <= 134:
    path = os.path.join(img_path, str(df_test.iloc[row, 0]))
    img = image.load_img(path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor_heatmap = img_tensor.astype('float32')
    img_tensor = np.expand_dims(img_tensor_heatmap, axis=0)
    img_tensor = preprocess_input(img_tensor)
    img_tensor /= 255
    pred = model.predict(img_tensor)
    print(pred)

    pred = model.predict(img_tensor)
    model_output = model.output[:, 0]
    last_conv_layer = model.get_layer('block5_conv3')
    grads = K.gradients(model_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img_tensor])
    for j in range(512):
        conv_layer_output_value[:, :, j] *= pooled_grads_value[j]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img_tensor.shape[2], img_tensor.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
    heatmap_array = image.img_to_array(heatmap)
    superimposed_img = heatmap_array * 0.25 + img_tensor_heatmap * 100
    f, axarr = plt.subplots(3, sharex='all')
    superimposed_img = image.array_to_img(superimposed_img)
    axarr[0].imshow(img)
    axarr[0].set_title(str(df_test.iloc[row, 0]) + '\n' + str(df_test.iloc[row, 7])
                       + '\n' + 'Predicted Label:' + str(np.argmax(pred) + 1) +
                       '\n' + 'Accuracy: ' + '%.2f' % float(df_test.iloc[row, 2]) + ' %' +
                       '\n' 'True Label: ' + str(df_test.iloc[row, 5]))
    axarr[2].imshow(superimposed_img)
    axarr[1].imshow(heatmap)
    accuracy = max(pred[0]) * 100
    plt.show()
    row += 1