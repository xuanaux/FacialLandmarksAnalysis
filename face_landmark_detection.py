# Based on example script from Dlib
import sys
import os
import cv2
import dlib
import glob
from skimage import io

from FacialLandmarks import FacialLandmarks


predictor_path = './shape_predictor_68_face_landmarks.dat'
faces_folder_path = './img'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
fl = FacialLandmarks(predictor_path)
# win = dlib.image_window()

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)

    # win.clear_overlay()
    # win.set_image(img)

    detections = detector(img, 1)
    print("Number of faces detected: {}".format(len(detections)))
    for k, d in enumerate(detections):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        landmarks = [p for p in shape.parts()]
        print("Is mouth open?: {}".format(fl.isMouthOpen(landmarks)))
        print("Is eye open?: {}".format(fl.isEyeOpen(landmarks)))

        for i in range(0,68): #There are 68 landmark points on each face
            #For each point, draw a red circle with thickness2 on the original frame
            cv2.circle(img, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) 

        # Draw the face landmarks on the screen.
        # win.add_overlay(shape)

    cv2.imshow("image", img) #Display the frame

    cv2.waitKey(0)

    # win.add_overlay(detections)
    # dlib.hit_enter_to_continue()
