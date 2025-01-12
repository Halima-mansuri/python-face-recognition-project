# pip install cmake
# pip install face-recognition
# pip install numpy
# pip install opencv
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)
# load known faces
fraz_image = face_recognition.load_image_file("faces/fraz.jpeg")
# create face encodings - to convert image to number - it makes easier to compare images
fraz_encodings = face_recognition.face_encodings(fraz_image)[0]
iqra_image = face_recognition.load_image_file("faces/iqra.jpeg")
# create face encodings - to convert image to number - it makes easier to compare images
iqra_encodings = face_recognition.face_encodings(iqra_image)[0]
# [0]-is used bcoz it returns a list of numbers of images ,but we want only iqra's image number

# now list the encodings and names of faces
known_face_encodings = [fraz_encodings, iqra_encodings]
known_face_names = ["fraz", "iqra"]

# list of expected students
students = known_face_names.copy()

# face ki locations h jisme se search kiya jayega sath sath use number me format bhi banaya jayega means encodings
face_locations = []
face_encodings = []

# get the current date and time to lock the time when the person is come and give it attendance
# ek variable me current time store karao
now = datetime.now()
# usko format me karao
current_date = now.strftime("%Y-%m-%d")

# create csv writer
# for which create a file
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

# main logic start here
while True:
    # video_capture.read returns two argument one=capturing is successfully or not and second=frame here we want only
    # frame argument
    _, frame = video_capture.read()
    # we want frame in small size so resize here
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # now convert text into rgb color
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # now it compare faces by encodings of images and detected faces in webcamera
    for face_encoding in face_encodings:
        # matches returns a true or false
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        # to check how much similar is the face_encoding with the known_face_encodings
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)

        # or jab distance minimum hoga tab sabse jyada similar hoga image to minimum value ki index nikalo
        best_match_index = np.argmin(face_distance)

        # condition
        if matches[best_match_index]:
            # agar true h to return kardo name jisse ki best match hua h
            name = known_face_names[best_match_index]
        # Add the text if a person is present
        if name in known_face_names:
            # set fonts WHEN WE WANT TO DISPLAY TEXT USING CV2
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + "Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

            # remove presented students from expected students
            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M:%S")
                # lnwriter me write karvana h row form me
                lnwriter.writerow([name, current_time])

    # image show karani h
    cv2.imshow("Attendance", frame)
    # single & is used bcoz waitkey is bitwise
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# file close camera is released and destroy all windows
video_capture.release()
cv2.destroyAllWindows()
f.close()
