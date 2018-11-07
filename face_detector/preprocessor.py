import os

import numpy as np 
import matplotlib.pyplot as plt
import cv2

from .person import Person


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
        x,y,w,h = contours[0]
        y -= 60
        h += 80
        return img[y:y+h,x:x+w]

    def load_image(self, filepath: str) -> np.ndarray:
        """
        Return a np.ndarray presentation of an image.
        If an invalid path is provided, it returns an
        empty np.array.

        :param filepath: path to a .jpg image
        """
        if os.path.isfile(filepath):    
            img_raw = cv2.imread(filepath)
            img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        else:
            img = np.array([])
        return img

    def get_face(self, img: np.ndarray) -> Person:
        """
        It finds a face in the image, and return a Person
        object, with the img and their features coordinates.

        :param img: a grayscale image.
        :return: a Person object, with the img and their features coordinates.
        """
        person = Person()
        img_copy = img.copy()
        #face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')
        face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
        nose_cascade = cv2.CascadeClassifier('cascades/Nariz.xml')
        faces = face_cascade.detectMultiScale(img_copy, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img_copy,(x,y),(x+w,y+h),(255,0,0),2)
            roi_color = img_copy[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_color)
            nose = nose_cascade.detectMultiScale(roi_color)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            for (ex,ey,ew,eh) in nose:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        person.face = self._crop_image(img, faces)
        person.face_with_contours = img_copy
        person.eyes = eyes
        return person
