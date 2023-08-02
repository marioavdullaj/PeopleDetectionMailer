from PIL import Image, ImageDraw as D
import cv2
import numpy as np
import datetime
from Detection import DetectorTransformer, MobileSSDDetector
from Mailer import Mailer
import uuid
import os
from dotenv import load_dotenv

# Global variables
start_time = datetime.datetime.now()
last_time_detected = datetime.datetime(2023,1,1)
sampling_time = 5
mailing_time = 30

# Environment variables
load_dotenv()
camera_path = os.getenv('CAMERA_PATH')
email_address = os.getenv("EMAIL_ACCOUNT")
app_password = os.getenv("EMAIL_PASSWORD")
email_receivers = [element.strip() for element in os.getenv("EMAIL_RECEIVERS").strip('[]').split(',')]

# MobileSSD model
#mobile_ssd_model_path = os.getenv('MOBILE_SSD_MODEL_PATH')
#dnn = MobileSSDDetector(path_to_ckpt=mobile_ssd_model_path)

# Transformer model
dnn = DetectorTransformer()

# Mailer class
mailer = Mailer(email_address, app_password, email_address, email_receivers)

# Camera stream capture via opencv
camera_path = os.getenv('CAMERA_PATH')
show_camera_preview = os.getenv('SHOW_CAMERA_PREVIEW') == "1"

if str.isnumeric(camera_path):
    vid = cv2.VideoCapture(int(camera_path))
else:
    vid = cv2.VideoCapture(camera_path)


while(True):
    end_time = datetime.datetime.now()
    ret, frame = vid.read()
    if frame is not None:
        image = cv2.resize(frame, (800,600))

    if (end_time-start_time).total_seconds() > sampling_time:
        start_time = end_time

        image, peopleDetected = dnn.processFrame(image)
        if(show_camera_preview):
            cv2.imshow("preview", image)

        if(peopleDetected):
            if((end_time-last_time_detected).total_seconds() > mailing_time):
                image_path = f"./Images/image_{uuid.uuid4()}.jpg"
                cv2.imwrite(image_path, frame)

                mailer.send("Detected people", "People detected at your home!", image_path)
                last_time_detected = end_time

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()