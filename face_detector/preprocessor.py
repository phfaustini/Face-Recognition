import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

from .person import Person
from .utils import Utils


class Preprocessor():

    """Perform low level tasks to turn .jpg images to more
    structured content."""

    def _crop_image(self, img: np.ndarray, contours: np.ndarray) -> np.ndarray:
        """
        It crops an image, given a set of delimiter points.

        :param img: a grayscale image.
        :param contours: a np.ndarray of points.
        :return: an np.ndarray image with only what is inside contours area.
        """
        try:
            x, y, w, h = contours[0]
            y -= 60
            h += 80
            return img[y:y+h, x:x+w]
        except:
            return None

    def _load_image(self, filepath: str) -> tuple:
        """
        Return a tuple with a np.ndarray presentation of an image
        and a label (class of the face).
        If an invalid path is provided, it returns an
        empty np.array.

        :param filepath: path to a .jpg image
        """
        if os.path.isfile(filepath):
            img_raw = cv2.imread(filepath)
            img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        else:
            img = np.array([])
        label = filepath.split('/')[1]
        return img, label

    def load_images(self) -> list:
        """It loads all faces images into a big list
        of (img, label), where img is a np.ndarray
        of the image and label is a string with the class
        of that image.

        :return: [(img: np.ndarray, label: str)]
        """
        return list(map(self._load_image, Utils.FILES))

    def get_face(self, img: np.ndarray, label: str) -> Person:
        """
        It finds a face in the image, and return a Person
        object, with the img and their features coordinates.

        :param img: a grayscale image.
        :paral label: class of that image (folder name)
        :return: a Person object, with the img and their features coordinates.
        """
        person = Person()
        img_copy = img.copy()
        face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')
        # face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
        nose_cascade = cv2.CascadeClassifier('cascades/Nariz.xml')
        faces = face_cascade.detectMultiScale(img_copy, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_color = img_copy[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_color)
            nose = nose_cascade.detectMultiScale(roi_color)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            for (ex, ey, ew, eh) in nose:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        person.face = self._crop_image(img, faces)
        if person.face is None:
            return None
        if person.face.shape[0] == 0:
            return None
        person.face_with_contours = img_copy
        person.eyes = eyes
        person.nose = nose
        person.original_img = img
        person.face_resized_notequalised = cv2.resize(person.face, (512, 512))
        person.face_resized = cv2.resize(cv2.equalizeHist(person.face), (512, 512))
        person.label = label
        return person
