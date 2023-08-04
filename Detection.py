from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw as D
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import time
import pandas

class MobileSSDDetector:
    def __init__(self, path="./Models/mobileSSDModel.pb", threshold=0.8):
        self.path_to_ckpt = path
        self.threshold = threshold
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        peopleDetected = False

        image_np_expanded = np.expand_dims(image, axis=0)
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
			int(boxes[0,i,1]*im_width),
			int(boxes[0,i,2] * im_height),
			int(boxes[0,i,3]*im_width))

        boxes, scores, classes =  boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()] #, int(num[0])

        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > self.threshold:
                box = boxes[i]
                cv2.rectangle(image,(box[1],box[0]),(box[3],box[2]),(134,235,52),2)
                cv2.rectangle(image, (box[1],box[0]-30),(box[1]+125,box[0]),(134,235,52), thickness=cv2.FILLED)
                cv2.putText(image, '  Person '+str(round(scores[i],2)), (box[1],box[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225,255,225), 1)
                peopleDetected = True

        return image, peopleDetected

    def close(self):
        self.sess.close()
        self.default_graph.close()


class DetectorTransformer:
    def __init__(self, threshold=0.8, image_size=640):
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.threshold = threshold
        self.image_size=image_size

    def __preprocessing(self,image):
        height, width, channels = image.shape
        scale = self.image_size / width
        image = cv2.resize(image, (self.image_size, (int)(height*scale)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        return image

    def processFrame(self, image):   
        peopleDetected = False 
        image = self.__preprocessing(image)
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=self.threshold)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            label = self.model.config.id2label[label.item()]
            if(label == "person"):
                peopleDetected = True
                print(f"Detected {label} with confidence {round(score.item(), 3)} at location {box}")
                draw=D.Draw(image)
                draw.rectangle([(box[0],box[1]),(box[2],box[3])],outline="red",width=2)
                draw.text((box[0], box[1]-10), label, fill ="red")

        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), peopleDetected

class YOLO:
    def __init__(self, yolo_repo="ultralytics/yolov5", yolo_path="./Models/yolov5s.pt", threshold=0.8, image_size=640):
        self.model = torch.hub.load(yolo_repo, "custom", path=yolo_path)
        self.threshold = threshold
        self.image_size = image_size

    def __preprocessing(self,image):
        height, width, channels = image.shape
        scale = self.image_size / width
        image = cv2.resize(image, (self.image_size, (int)(height*scale)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        return image
    
    def processFrame(self, image):
        peopleDetected = False
        image = self.__preprocessing(image)
        
        outputs = self.model([image], size=self.image_size)
        
        results = outputs.pandas().xyxy[0]
        dt = pandas.DataFrame(results)
        for row in dt.index:
            score, label, box = dt['confidence'][row], dt['name'][row], [dt['xmin'][row], dt['ymin'][row], dt['xmax'][row], dt['ymax'][row]]
            box = [round(i, 2) for i in box]
            if(label == "person"):
                peopleDetected = True
                print(f"Detected {label} with confidence {round(score.item(), 3)} at location {box}")
                draw=D.Draw(image)
                draw.rectangle([(box[0],box[1]),(box[2],box[3])],outline="red",width=2)
                draw.text((box[0], box[1]-10), label, fill ="red")
        
        return cv2.cvtColor(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB), peopleDetected

