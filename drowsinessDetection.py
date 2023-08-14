import dlib
import cv2
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import imutils
import urllib.request
import time

# ThingSpeak
t = time.time()
val1 = val2 = 0
URl = "https://api.thingspeak.com/update?api_key="
KEY = "T2XVWXFK0HXMPHR9"


# Variables
eyeTresh = 0.30
eyeFrameCheck = 8
eyeFrameCount = 0
mouthTresh = 0.70
mouthFrameCheck = 5
mouthFrameCount = 0
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def eyeAspectRatio(eye):
    v1 = distance.euclidean(eye[1], eye[5])
    v2 = distance.euclidean(eye[2], eye[4])
    h = distance.euclidean(eye[0], eye[3])
    ear = (v1 + v2) / (2 * h)
    return ear


def mouthAspectRatio(mouth):
    v1 = distance.euclidean(mouth[1], mouth[11])
    v2 = distance.euclidean(mouth[2], mouth[10])
    v3 = distance.euclidean(mouth[3], mouth[9])
    v4 = distance.euclidean(mouth[4], mouth[8])
    v5 = distance.euclidean(mouth[5], mouth[7])
    h = distance.euclidean(mouth[0], mouth[6])
    mar = (v1 + v2 + v3 + v4 + v5) / (5 * h)
    return mar


if __name__ == "__main__":
    face_detector = dlib.get_frontal_face_detector()
    landmarkPredictor = dlib.shape_predictor(
        "model/shape_predictor_68_face_landmarks.dat"
    )

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("assets/testVideo.mp4")
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector(grayFrame)
        # print(face)
        for face in faces:
            faceLandmarks = landmarkPredictor(grayFrame, face)
            faceLandmarks_np = face_utils.shape_to_np(faceLandmarks)

            leftEye, rightEye = (
                faceLandmarks_np[lStart:lEnd],
                faceLandmarks_np[rStart:rEnd],
            )
            # eye
            leftEAR = eyeAspectRatio(leftEye)
            rightEAR = eyeAspectRatio(rightEye)
            ear = (leftEAR + rightEAR) / 2
            # print(f"ear:{ear}")
            # mouth
            mouth = faceLandmarks_np[49:61]
            mar = mouthAspectRatio(mouth)
            # print(f"mar:{mar}")

            # drowsiness detection
            print()
            if mar > mouthTresh:
                mouthFrameCount += 1
                if mouthFrameCount >= mouthFrameCheck:
                    print("Yawning : ", end="")
            else:
                mouthFrameCount = 0
                print("Not Yawning : ", end="")

            if ear <= eyeTresh:
                eyeFrameCount += 1
                if eyeFrameCount >= eyeFrameCheck:
                    print("Sleepy", end="")
                else:
                    print("Not Sleepy", end="")  # in this line we can record blink
            else:
                eyeFrameCount = 0
                print("Not Sleepy", end="")

            # Plot eye in frame
            faceFeatures = np.concatenate((leftEye, rightEye, mouth))
            for pt in faceFeatures:
                cv2.circle(frame, pt, 1, (0, 255, 255), 1)

            # To print all 69 landmarks
            # for n in range(0, 68):
            #     x = faceLandmarks.part(n).x
            #     y = faceLandmarks.part(n).y
            #     cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

        cv2.imshow("Face Landmarks", frame)
        cv2.imshow("Grayscale Frame", grayFrame)

        key = cv2.waitKey(1)
        if key is ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
