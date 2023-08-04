# People Detection Mailer
A surveillance python app.
This app uses people detection deep neural network models over a stream of images acquired through a camera, and automatically sends emails whenever people are detected. The emails also include images as attachments.

Currently it uses the following object detection models:
- MobileNet SSD
- DETR: Detection Transformer
- YOLOv5

## Installing / Getting started

First you need to install all the python requirements. To do so, execute the following command:

```shell
pip install -r requirements.txt
```

After the installation is complete, just execute the python script

```shell
python main.py
```

Make sure to setup correctly the .env file.