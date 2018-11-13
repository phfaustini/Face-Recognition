from face_detector.preprocessor import Preprocessor
from face_detector.transformer import Transformer
from face_detector.model import Model
from face_detector.utils import Utils

if __name__ == "__main__":
    p = Preprocessor()
    u = Utils()
    m = Model()

    data = []
    imgs = p.load_images()  # [(img: np.ndarray, label: str)]
    for img in imgs:
        person = p.get_face(img=img[0], label=img[1])
        if person is not None:
            data.append((person.face_resized, person.label))
    # person = p.get_face(img=imgs[130][0], label=imgs[130][1])
    # u.display_image(person.face_resized)
    m.classify(data=data)
