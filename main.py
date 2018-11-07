from face_detector.preprocessor import Preprocessor
from face_detector.utils import Utils

if __name__ == "__main__":
    p = Preprocessor()
    u = Utils()
    img = p.load_image('faces/image_0074.jpg')
    person = p.get_face(img)
    print(person.eyes)
    u.display_image(person.face_with_contours)
