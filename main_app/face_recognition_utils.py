# face_recognition_utils.py
import face_recognition

def find_similar_faces(photo_encoding, known_face_encodings, known_face_paths):
    similar_face_paths = []

    for i, known_encoding in enumerate(known_face_encodings):
        res = face_recognition.compare_faces([known_encoding], photo_encoding)
        if res[0]:
            similar_face_paths.append(known_face_paths[i])

    return similar_face_paths
