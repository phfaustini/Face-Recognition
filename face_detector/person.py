import numpy as np


class Person():

    """Representation of a person, with their features coordinates."""

    def __init__(self):
        self.face = np.array([])
        self.face_with_contours = np.array([])
        self.eyes = np.array([])
