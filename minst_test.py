import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=200)
image_data = cv2.imread('fashion_mnist_images/train/4/0011.png', cv2.IMREAD_UNCHANGED)

print(image_data)
plt.imshow(image_data, cmap='gray')
plt.show()