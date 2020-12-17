# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from cv2 import cv2
import numpy as np
import face_recognition


def load_faces(img_path):
    # face_images = []
    images = []
    names = []
    files = os.listdir(img_path)
    for file in files:
        images.append(face_recognition.load_image_file(img_path + '/' + file))
        names.append(file[:-4])
    image_dict = dict(zip(names, images))
    return image_dict


def encode_faces(image_dict):
    for key in image_dict:
        image_dict[key] = face_recognition.face_encodings(image_dict[key])[0]
    return image_dict


# def compareFaces(image_path, known_faces_dict):
#     unknown_image = face_recognition.load_image_file(image_path)
#     unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
#     results = face_recognition.compare_faces(known_faces, unknown_face_encoding)
#     print(results)


if __name__ == "__main__":

    # open the camera
    video_capture = cv2.VideoCapture(1)

    # load sample pictures and learn how to recognize it
    images_dict = load_faces("./images")

    # create a dictionary of known face encodings and their names
    known_faces_dict = encode_faces(images_dict)

    # initialize some variable
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        # grab a single frame of video
        ret, frame = video_capture.read()

        # resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # only process every other frame of video to save time
        if process_this_frame:
            # find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

        for face_encoding in face_encodings:
            # see if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(list(known_faces_dict.values()), face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(list(known_faces_dict.values()), face_encoding)
            best_match_index = int(np.argmin(face_distances))
            if matches[best_match_index]:
                name = list(known_faces_dict)[best_match_index]
            face_names.append(name)

        process_this_frame = not process_this_frame

        # display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # display the resulting image
        cv2.imshow('Video', frame)

        # hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
