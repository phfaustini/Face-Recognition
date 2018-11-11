import numpy as np


class Person():

    """Representation of a person, with their features coordinates."""

    def __init__(self):
        self.original_img = np.array([])
        self.face = np.array([])
        self.face_with_contours = np.array([])
        self.eyes = np.array([])
        self.nose = np.array([])
        self.face_resized = np.array([])
        self.label = None
