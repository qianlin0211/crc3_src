#!/usr/bin/env python
import tensorflow as tf
from cv_bridge import CvBridge, CvBridgeError
import rospy

import os
import sys
import cv2
import numpy as np
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image


# Set model here ############
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# By default models are stored in data/models/
MODEL_PATH = os.path.join(
    os.path.dirname(sys.path[0]), 'data', 'models', MODEL_NAME)
# Path to frozen detection graph. This is the actual model that is used
# for the object detection.
PATH_TO_CKPT = MODEL_PATH + '/frozen_inference_graph.pb'
# Set the label map file here ###########
LABEL_NAME = 'sign_label.pbtxt'
# By default label maps are stored in data/labels/
PATH_TO_LABELS = os.path.join(
    os.path.dirname(sys.path[0]), 'data', 'labels', LABEL_NAME)
# Set the number of classes here #########
NUM_CLASSES = 4


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`,
# we know that this corresponds to `airplane`.  Here we use internal utility functions,
# but anything that returns a dictionary mapping integers to appropriate
# string labels would be fine


class Detector:

    def __init__(self):
        self.image_pub = rospy.Publisher("debug_image", Image, queue_size=1)
        # self.object_pub = rospy.Publisher(
         #   "objects", Detection2DArray, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/kinect2/qhd/image_color_rect", Image, self.image_cb, queue_size=1, buff_size=2**24)
        self.image_pub = rospy.Publisher(
            'masked_image', Image, queue_size=1)
        self.sess = tf.Session(graph=detection_graph)

    def image_cb(self, data):
        # objArray = Detection2DArray()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        det_image = np.copy(image)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = np.asarray(image)
        # Expand dimensions since the model expects images to have shape: [1,
        # None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was
        # detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.sess.run([
            boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        class_names = ['BG', 'left', 'right', 'forward', 'stop']

        det_image = self.display_instances(
            det_image, boxes[0], classes[0],
                          class_names, scores[0])
       # for i in range(num_detections):
       #     class_id = int(classes[0][i])
        # print(boxes[0])
        marked_image_msg = self.bridge.cv2_to_imgmsg(det_image, 'rgb8')
        self.image_pub.publish(marked_image_msg)

    def random_colors(self, N):
        np.random.seed(1)
        colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
        return colors

    def display_instances(self, image, boxes, ids, names, scores):
        n_instances = boxes.shape[0]
        if not n_instances:
            print('No instances to display')
        else:
            assert boxes.shape[0] == ids.shape[0]

        colors = self.random_colors(n_instances)
        height, width = image.shape[:2]

        for i, color in enumerate(colors):
            if not np.any(boxes[i]):
                continue
            if scores[i] < 0.7:
                break

            y1 = int(boxes[i][0] * image.shape[0])
            x1 = int(boxes[i][1] * image.shape[1])
            y2 = int(boxes[i][2] * image.shape[0])
            x2 = int(boxes[i][3] * image.shape[1])
            # print(y2)

            # mask = masks[:,:, i]
            # image = apply_mask(image, mask, color)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (125, 255, 50), 2)

            label = names[int(ids[i])]
            score = scores[i] if scores is not None else None

            caption = '{} {:.2f}'.format(label, score) if score else label
            image = cv2.putText(
                image, label, (
                    x1, y1 - 7), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2
            )

        return image

if __name__ == '__main__':
    rospy.init_node('detector_node')
    obj = Detector()
    rospy.spin()
