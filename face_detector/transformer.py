import numpy as np
import matplotlib.pyplot as plt
import cv2

class Transformer():

    """This class converts images to structured
    data, by extracting features from images.
    """
    
    def img2histogram(self, img: np.ndarray) -> np.ndarray:
        equalised_img = cv2.equalizeHist(img)
        histogram = np.zeros(256, dtype=int)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                histogram[img[i][j]] += 1
        return histogram