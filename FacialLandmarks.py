import cv2
import dlib
import numpy as np

'''
68 Landmarks components:
0  - 16: Jaw line
17 - 21: Left eyebrow
22 - 26: Right eyebrow
27 - 30: Nose bridge
30 - 35: Lower nose
36 - 41: Left eye
42 - 47: Right Eye
48 - 59: Outer lip
60 - 67: Inner lip
'''

class FacialLandmarks:
    def __init__(self, facePredictor):
        assert facePredictor is not None

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(facePredictor)
        self.ratioThresh = 0.1

    def getAllFaceBoundingBoxes(self, rgbImg):
        """
        Find all face bounding boxes in an image.
        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :return: All face bounding boxes in an image.
        :rtype: dlib.rectangles
        """
        assert rgbImg is not None

        try:
            return self.detector(rgbImg, 1)
        except Exception as e:
            print("Warning: {}".format(e))
            # In rare cases, exceptions are thrown.
            return []

    def getLargestFaceBoundingBox(self, rgbImg, skipMulti=False):
        """
        Find the largest face bounding box in an image.
        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param skipMulti: Skip image if more than one face detected.
        :type skipMulti: bool
        :return: The largest face bounding box in an image, or None.
        :rtype: dlib.rectangle
        """
        assert rgbImg is not None

        faces = self.getAllFaceBoundingBoxes(rgbImg)
        if (not skipMulti and len(faces) > 0) or len(faces) == 1:
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            return None

    def findLandmarks(self, rgbImg, bb):
        """
        Find the landmarks of a face.
        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param bb: Bounding box around the face to find landmarks for.
        :type bb: dlib.rectangle
        :return: Detected landmark locations.
        :rtype: list of (x,y) tuples
        """
        assert rgbImg is not None
        assert bb is not None

        points = self.predictor(rgbImg, bb)
        return [p for p in points.parts()]

    def isMouthOpen(self, landmarks):
        mouth = landmarks[60:68]
        verticalDist = abs(mouth[1].y + mouth[2].y + mouth[3].y - mouth[5].y - mouth[6].y - mouth[7].y) / 3
        horizontalDist = abs(mouth[4].x - mouth[0].x)
        return 1.0 * verticalDist / horizontalDist > self.ratioThresh

    def isEyeOpen(self, landmarks):
        leftEye  = landmarks[36:42]
        rightEye = landmarks[42:48]
        leftEyeVerticalDist     = abs(leftEye[1].y + leftEye[2].y - leftEye[4].y - leftEye[5].y) / 2
        leftEyeHorizontalDist   = abs(leftEye[3].x - leftEye[0].x)
        rightEyeVerticalDist    = abs(rightEye[1].y + rightEye[2].y - rightEye[4].y - rightEye[5].y) / 2
        rightEyeHorizontalDist  = abs(rightEye[3].x - rightEye[0].x)
        return (1.0 * leftEyeVerticalDist / leftEyeHorizontalDist > self.ratioThresh or 
                1.0 * rightEyeVerticalDist / rightEyeHorizontalDist > self.ratioThresh)

