*************
Face Detector
*************

This project uses supervised learning to identify who is who
in images of faces.

First, faces in images are identified and cropped. A
histogram equalisation is performed, and the resulting image is
scaled to 512x512 pixels. Then, it is passed to a PCA
algorithm to reduce dimensionality (each pixel is a feature).

Finally, different classification algorithms are used. Models are 
trained cross-validation, with 70% of the dataset used for training, and the remainig is used for testing. 
The goal is to provide and image to the model, and the model must 
say who is the person in that image.

Environment used:
-----------------
* `Python3.7 + Anaconda <https://www.anaconda.com/download/#linux>`_
* See `requirements.txt <requirements.txt>`_


Structure:
----------

* **face_detector/** - Source code.
* **faces/** - .jpg files inside A-X folders. Dataset used: http://www.vision.caltech.edu/Image_Datasets/faces/faces.tar
* **cascades/** - .xml cascades files for feature recognition. See more in http://alereimondo.no-ip.org/OpenCV/34
* **experiments.ipynb** - jupyter notebook with results.
* **main.py** - Just type :code:`python main` to run experiments.

For copyright issues, I do not make pictures used publicly available. However, they can be found at http://www.vision.caltech.edu/Image_Datasets/faces/faces.tar.
Their organisation into *faces* folder follows the folder structure according to file *face_detector/utils.py*.
