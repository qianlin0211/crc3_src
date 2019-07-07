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
import message_filters
from crc3_perception.msg import detection

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

        # self.object_pub = rospy.Publisher(
         #   "objects", Detection2DArray, queue_size=1)
        self.bridge = CvBridge()
        self.color_image_sub = message_filters.Subscriber(
            "/kinect2/qhd/image_color_rect", Image)
        self.depth_image_sub = message_filters.Subscriber(
            "/kinect2/qhd/image_depth_rect", Image)
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.color_image_sub, self.depth_image_sub], queue_size=5, slop=0.1)
        self.synchronizer.registerCallback(self.callback)
        self.image_pub = rospy.Publisher(
            'detected_image', Image, queue_size=1)
        self.detect_pub = rospy.Publisher(
            'perception', detection, queue_size=1)
        self.sess = tf.Session(graph=detection_graph)

    def callback(self, color_msg, depth_msg):
        # objArray = Detection2DArray()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg) / 1000.0
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
                          class_names, scores[0], depth_image)
       # for i in range(num_detections):
       #     class_id = int(classes[0][i])
        # print(boxes[0])
        marked_image_msg = self.bridge.cv2_to_imgmsg(det_image, 'rgb8')
        self.image_pub.publish(marked_image_msg)

    def random_colors(self, N):
        np.random.seed(1)
        colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
        return colors

    def display_instances(self, image, boxes, ids, names, scores, depth_image):
        n_instances = boxes.shape[0]
        if not n_instances:
            print('No instances to display')
        else:
            assert boxes.shape[0] == ids.shape[0]

        colors = self.random_colors(n_instances)
        height, width = image.shape[:2]

        detect = detection()

        # print(dis)
        const_dis = 10000.0
        stop_dis = 10000.0
        detect.stop_sign_found = False
        detect.dist_to_stop = 0.0
        detect.direction = 'NONE'
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
            dis = self.depth_find(y1, x1, y2, x2, image, depth_image)
            if int(ids[i]) == 4 and dis < stop_dis and dis > 0.0:
                detect.stop_sign_found = True
                detect.dist_to_stop = dis
                stop_dis = dis
            if dis < const_dis and int(ids[i]) != 4:
                detect.direction = names[int(ids[i])]
                const_dis = dis

            if dis > 0.0:

                cv2.putText(image,  '%.2f m' % dis, (x1, y1 - 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (50, 255, 255), 2)

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
        self.detect_pub.publish(detect)

        return image

    def depth_find(self, y1, x1, y2, x2, cv_image, cv_depth_image):

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        depth = 0
        count = 0
        for i in range(abs(x2 - x1) / 2):
            for j in range(abs(y2 - y1) / 2):
                x = i + min(x1, x2) + abs(x2 - x1) / 4
                y = j + min(y1, y2) + abs(y2 - y1) / 4

                if float(cv_depth_image[y, x]) > 0:
                    depth += float(cv_depth_image[y, x])
                    count += 1
        if count == 0:
            depth = float(cv_depth_image[cy, cx])
        else:
            depth = depth / count

        return depth

if __name__ == '__main__':
    rospy.init_node('detector_node')
    obj = Detector()
    rospy.spin()
