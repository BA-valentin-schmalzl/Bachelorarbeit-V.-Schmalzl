from keras.preprocessing import image
import numpy as np
from scipy import signal
import skimage.measure
import cv2

path = 'Bildpfad'
img = image.load_img(path, target_size=(1200, 1600))
img_t = image.img_to_array(img)
img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
img_t = np.expand_dims(img_t, axis=2)
img_t = img_t[400:800, 300:900]
img = image.array_to_img(img_t)
img.show()

FT1 = np.array([[0, 0, 0],
               [-1, 1, 0],
               [0, 0, 0]])
FT2 = np.array([[0, -1, 0],
               [0, 1, 0],
               [0, 0, 0]])
FT1 = FT1[:, :, None]
FT2 = FT2[:, :, None]
print(np.shape(FT1))
print(np.shape(img_t))
grad1 = signal.convolve(img_t, FT1, mode='same')
grad2 = signal.convolve(img_t, FT2, mode='same')
Pool = np.squeeze(grad1, axis=2)
Pool = skimage.measure.block_reduce(Pool, (2, 2), np.max)
Kontrast = cv2.convertScaleAbs(grad1, alpha=1, beta=0)
Kontrast = np.expand_dims(Kontrast, axis=2)
Pool = cv2.convertScaleAbs(Pool, alpha=1, beta=0)
Pool = np.expand_dims(Pool, axis=2)
Pool = image.array_to_img(Pool)
Pool.show()
Kontrast = image.array_to_img(Kontrast)
Kontrast.show()
Kontrast = cv2.convertScaleAbs(grad2, alpha=1, beta=0)
Kontrast = np.expand_dims(Kontrast, axis=2)
Kontrast = image.array_to_img(Kontrast)
Kontrast.show()
