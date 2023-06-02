import cv2
import time

# Load the face and eye cascades from the OpenCV library. This will be the cornerstone of this software.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load the video footage into the IDE. Realistically, the user should have to upload it through a GUI but it was not implemented.
cap = cv2.VideoCapture("test1.mp4")

# Initialization of variables
frame_count = 0
last_increment_time = time.time()
eye_contact_count = 0
face_rectangles = []
eye_rectangles = []

while True:
    # Infinite loop that keeps reading frames from the video footage until there are no more frames
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find all faces in the frame, with the use of pythons OpenCV library
    face_rectangles = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over each face
    for face_rect in face_rectangles:
        # Find all eyes within the face rectangle
        x, y, w, h = face_rect
        eye_rectangles = eye_cascade.detectMultiScale(gray[y:y+h, x:x+w], scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Check if both eyes are within the face rectangle
        if len(eye_rectangles) == 2 and all([x > 0 and y > 0 and x + w < frame.shape[1] and y + h < frame.shape[0] for x, y, w, h in eye_rectangles]):
            # Check if 1 second has passed since the last increment
            current_time = time.time()
            if current_time - last_increment_time >= 1:
                # Increment the eye contact count and update the last increment time
                eye_contact_count += 1
                last_increment_time = current_time

    # Increment the frame count
    frame_count += 1

    # Show the video feed with the eye contact count
    cv2.putText(frame, "Eye contact count: {}".format(eye_contact_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Eye Tracking", frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('w'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
