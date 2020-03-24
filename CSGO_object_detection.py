import time
import cv2
import mss
import numpy as np
import os
import sys

import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO

# title of our window
title = "FPS benchmark"
start_time = time.time()
display_time = 2
fps = 0
sct = mss.mss()
width = 800
height = 640
monitor = {"top": 40, "left": 0, "width": width, "height": height}

# ## Env setup
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of 
tf.gfile = tf.io.gfile

# # Model preparation
PATH_TO_FROZEN_GRAPH = 'CSGO_inference_graph/CSGO_frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'CSGO_training/CSGO_labelmap.pbtxt'
NUM_CLASSES = 4

# ## Load a (frozen) Tensorflow model into memory.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# # Detection
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            # Get raw pixels from the screen, save it to a Numpy array
            image_np = np.array(sct.grab(monitor))
            # To get real color we do this:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Visualization of the results of a detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=3)

            # Counter Terrorist & Terrorist Information
            # CH (Counter Terrorist Head)
            array_ch = []
            for i, b in enumerate(boxes[0]):
                if classes[0][i] == 2: # ch
                    if scores[0][i] >= 0.5:
                        mid_x = ((boxes[0][i][1]) + boxes[0][i][3]) / 2
                        mid_y = ((boxes[0][i][0]) + boxes[0][i][2]) / 2
                        array_ch.append([mid_x, mid_y])
                        cv2.circle(image_np, (int(mid_x*width), (mid_y*height)), 3, (0, 0, 255), -1)

            # C (Counter Terrorist)
            array_c = []
            for i, b in enumerate(boxes[0]):
                if classes[0][i] == 1:  # c
                    if scores[0][i] >= 0.5:
                        mid_x = ((boxes[0][i][1]) + boxes[0][i][3]) / 2
                        mid_y = ((boxes[0][i][0]) + boxes[0][i][2] - boxes[0][i][0]) / 6
                        array_c.append([mid_x, mid_y])
                        cv2.circle(image_np, (int(mid_x * width), (mid_y * height)), 3, (50, 150, 255), -1)

            # TH (Terrorist Head)
            array_th = []
            for i, b in enumerate(boxes[0]):
                if classes[0][i] == 2:  # th
                    if scores[0][i] >= 0.5:
                        mid_x = ((boxes[0][i][1]) + boxes[0][i][3]) / 2
                        mid_y = ((boxes[0][i][0]) + boxes[0][i][2]) / 2
                        array_th.append([mid_x, mid_y])
                        cv2.circle(image_np, (int(mid_x * width), (mid_y * height)), 3, (0, 0, 255), -1)

            # T (Terrorist)
            array_t = []
            for i, b in enumerate(boxes[0]):
                if classes[0][i] == 1:  # c
                    if scores[0][i] >= 0.5:
                        mid_x = ((boxes[0][i][1]) + boxes[0][i][3]) / 2
                        mid_y = ((boxes[0][i][0]) + boxes[0][i][2] - boxes[0][i][0]) / 6
                        array_t.append([mid_x, mid_y])
                        cv2.circle(image_np, (int(mid_x * width), (mid_y * height)), 3, (50, 150, 255), -1)

            # Show image with detection
            cv2.imshow(title, cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            # Bellow we calculate our FPS
            fps += 1
            TIME = time.time() - start_time
            if (TIME) >= display_time:
                print("FPS: ", fps / (TIME))
                fps = 0
                start_time = time.time()
            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
