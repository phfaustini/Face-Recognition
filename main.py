from face_detector.preprocessor import Preprocessor
from face_detector.transformer import Transformer
from face_detector.model import Model
from face_detector.utils import Utils

if __name__ == "__main__":
    p = Preprocessor()
    t = Transformer()
    u = Utils()
    m = Model()
    
    imgs = p.load_images() #  [(img: np.ndarray, label: str)]
    #person = p.get_face(img=imgs[56][0], label=imgs[56][1])
    #print(person.label)
    #u.display_image(person.face_resized)
    m.classify(data=imgs)
