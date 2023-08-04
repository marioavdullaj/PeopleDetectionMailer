print("Loading libraries...")
from PIL import Image, ImageDraw as D
import cv2
import numpy as np
import datetime
from Detection import DetectorTransformer, MobileSSDDetector, YOLO
from Mailer import Mailer
import uuid
import os
from dotenv import load_dotenv
print("DONE!")

def initializeCamera(camera_path):
    print("Initializing videocamera stream....")
    if str.isnumeric(camera_path):
        v = cv2.VideoCapture(int(camera_path))
    else:
        v = cv2.VideoCapture(camera_path)
    print("DONE!")
    return v

# Global variables
start_time = datetime.datetime.now()
last_time_detected = datetime.datetime(2023,1,1)

# Environment variables
print("Loading environment parameters....")
load_dotenv()
camera_path = os.getenv('CAMERA_PATH')
email_address = os.getenv("EMAIL_ACCOUNT")
app_password = os.getenv("EMAIL_PASSWORD")
email_receivers = [element.strip() for element in os.getenv("EMAIL_RECEIVERS").strip('[]').split(',')]
sampling_time = float(os.getenv('SAMPLING_TIME'))
mailing_time = int(os.getenv('MAILING_TIME'))
print("DONE!")

print("Loading detection model....")
# Available DNN models
#dnn = DetectorTransformer()
#dnn = MobileSSDDetector()
dnn = YOLO()
print("DONE!")

# Mailer class
print("Mailing handler setup...")
mailer = Mailer(email_address, app_password, email_address, email_receivers)
print("DONE!")

# Camera stream capture via opencv
camera_path = os.getenv('CAMERA_PATH')
show_camera_preview = os.getenv('SHOW_CAMERA_PREVIEW') == "1"

cameraInitialized = False
while(not cameraInitialized):
    try:
        vid = initializeCamera(camera_path)
        cameraInitialized = vid is not None
    except:
        print("Failed to inizialize the camera! Retrying...")
        cameraInitialized = False

print("The application is running....")
while(True):
    end_time = datetime.datetime.now()
    try:
        ret, frame = vid.read()
        if ret:
            if (end_time-start_time).total_seconds() > sampling_time:
                start_time = end_time

                image, peopleDetected = dnn.processFrame(frame)
                if(show_camera_preview):
                    cv2.imshow("preview", image)

                if(peopleDetected):
                    if((end_time-last_time_detected).total_seconds() > mailing_time):
                        if not os.path.exists("./Images"):
                            os.makedirs("./Images")
                            
                        image_path = f"./Images/image_{uuid.uuid4()}.jpg"
                        cv2.imwrite(image_path, frame)

                        mailer.send("Detected people", "People detected at your home!", image_path)
                        last_time_detected = end_time
    except:
        print("Failed to capture the frame from the image stream\nReinitializing the videocamera stream...")
        vid = initializeCamera(camera_path)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()