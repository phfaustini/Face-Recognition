import numpy as np 
import matplotlib.pyplot as plt
import cv2


class Utils():

    """Class with auxiliary methods."""
    

    def display_image(self, img: np.ndarray):
        """Display an image until window is closed."""
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyallwindows()
