from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import time
import urllib.request

app = Flask(__name__)
cap = cv2.VideoCapture(0)

# ThingSpeak
t = time.time()
URl = "https://api.thingspeak.com/update?api_key="
KEY = "T2XVWXFK0HXMPHR9"

# Model Variables
eyeStatus = 0
yawnStatus = 0
eyeTresh = 0.31
eyeFrameCheck = 8
eyeFrameCount = 0
mouthTresh = 0.65
mouthFrameCheck = 5
mouthFrameCount = 0
# driverStatus = ""
featureFlag = False
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def extractFrames():
    while True:
        success, frame = cap.read()
        if not success:
            print("[*] Unable to access webcam.")
            break
        else:
            # Drowsiness detection
            faceFeatures = fatigueDetection(frame)
            if featureFlag == True:
                for pt in faceFeatures:
                    cv2.circle(frame, pt, 1, (0, 255, 255), 1)

            frame = cv2.putText(
                frame,
                f"Eye Closed: {eyeStatus}",
                (25, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            frame = cv2.putText(
                frame,
                f"Yawning: {yawnStatus}",
                (25, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            if yawnStatus == 1 and eyeStatus == 0:
                frame = cv2.putText(
                    frame,
                    "Driver is drowsy",
                    (25, 400),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif eyeStatus == 1:
                frame = cv2.putText(
                    frame,
                    "Driver is sleepy!",
                    (25, 400),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                frame = cv2.putText(
                    frame,
                    "Active",
                    (25, 400),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Stream on webapp
            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


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


def fatigueDetection(frame):
    global yawnStatus
    global eyeStatus
    global driverStatus
    global eyeFrameCount
    global mouthFrameCount
    global t
    global featureFlag

    driverStatus = ""
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
        if mar > mouthTresh:
            mouthFrameCount += 1
            if mouthFrameCount >= mouthFrameCheck:
                # print("Yawning : ", end="")
                yawnStatus = 1
                # driverStatus += "Yawning + "
        else:
            mouthFrameCount = 0
            yawnStatus = 0
            # print("Not Yawning : ", end="")
            # driverStatus += "Not Yawning + "

        if ear <= eyeTresh:
            eyeFrameCount += 1
            if eyeFrameCount >= eyeFrameCheck:
                # print("Sleepy", end="")
                eyeStatus = 1
                # driverStatus += "Sleepy"
            else:
                # print("Not Sleepy", end="")  # in this line we can record blink
                # driverStatus += "Not Sleepy"
                eyeStatus = 0
        else:
            eyeFrameCount = 0
            eyeStatus = 0
            # print("Not Sleepy", end="")
            # driverStatus += "Not Sleepy"
    # print(driverStatus)
    if (time.time() - t) > 10:
        HEADER = "&field1={}&field2={}".format(eyeStatus, yawnStatus)
        NEW_URL = URl + KEY + HEADER
        data = urllib.request.urlopen(NEW_URL)
        t = time.time()
        print("[*] Uploaded: ", data)
    try:
        featureFlag = True
        return np.concatenate((leftEye, rightEye, mouth))
    except:
        featureFlag = False
        return 0


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/stream")
def stream():
    return Response(
        extractFrames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    face_detector = dlib.get_frontal_face_detector()
    landmarkPredictor = dlib.shape_predictor(
        "model/shape_predictor_68_face_landmarks.dat"
    )
    app.run(debug=True)
