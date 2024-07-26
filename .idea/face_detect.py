import cv2
import cv2.data

face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + r"haarcascade_frontalface_default.xml")
body_cap = cv2.CascadeClassifier(cv2.data.haarcascades + r"haarcascade_fullbody.xml")
upper_body_cap = cv2.CascadeClassifier(cv2.data.haarcascades + r"haarcascade_upperbody.xml")
eye_cap = cv2.CascadeClassifier(cv2.data.haarcascades + r"haarcascade_eye.xml")
profileface_cap = cv2.CascadeClassifier(cv2.data.haarcascades + r"haarcascade_profileface.xml")
video_cap = cv2.VideoCapture(1)

while True:
    ret , video_data = video_cap.read()
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    faces =  face_cap.detectMultiScale(
        col,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30)
    )
    body = body_cap.detectMultiScale(
        col,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30)
        )
    upper_body = upper_body_cap.detectMultiScale(
        col,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30)
        )
    eyes = eye_cap.detectMultiScale(
        col,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30)
        )
    profileface = profileface_cap.detectMultiScale(
        col,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30)
        )
    for (x,y,w,h) in faces:
        cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),1)
    cv2.imshow("Video_live", video_data)
    if cv2.waitKey(10) == ord('q'):
        break
video_cap.release()