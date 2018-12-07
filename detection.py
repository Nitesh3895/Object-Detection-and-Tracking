"""
Application: Object Detection.
Author: Nitesh Kumar M
Creation Date: 05/12/2018
"""
# pylint: disable=E0611
from __future__ import print_function
import logging
import tensorflow as tf
from numpy import asarray, expand_dims, squeeze
from cv2 import cvtColor, COLOR_RGB2BGR, COLOR_BGR2RGB

face_crop_list = []

class Detection:
    def __init__(self):

        #Bounding Box size to extend
        self.extend_ratio_w = 0.1
        self.extend_ratio_h = 0.1
        #Minimum threshold probability of a face to be detected
        self.score_threshold = 0.50
        # load graph and set required fields
        self.frozen_graph = self.load_frozen_graph('./models/frozen_inference_graph.pb')
        self.image_tensor = self.frozen_graph.get_tensor_by_name('detection/image_tensor:0')
        self.boxes = self.frozen_graph.get_tensor_by_name('detection/detection_boxes:0')
        self.scores = self.frozen_graph.get_tensor_by_name('detection/detection_scores:0')
        self.classes = self.frozen_graph.get_tensor_by_name('detection/detection_classes:0')
        self.num_detections = self.frozen_graph.get_tensor_by_name('detection/num_detections:0')
        self.sess = tf.Session(graph=self.frozen_graph)

    def load_frozen_graph(self, frozen_graph_filename):
            # We load the protobuf file from the disk and parse it to retrieve the
            # unserialized graph_def
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            # Then, we can use again a convenient built-in function to import a graph_def into the
            # current default Graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="detection",
                producer_op_list=None
            )
        return graph


    def detect_object(self, image_path):
        count = 0
        # Read & convert jpeg image into numpy array
        image_input = image_path
        image_input_tmp = cvtColor(image_input, COLOR_RGB2BGR)
        image_np = asarray(image_input_tmp)

        im_height, im_width, im_channels = image_np.shape
        image_np_expanded = expand_dims(image_np, axis=0) 

        bo, scos, clas, det = self.sess.run([self.boxes, self.scores, self.classes, \
            self.num_detections], feed_dict={self.image_tensor: image_np_expanded})
        scos = squeeze(scos)
        bo = squeeze(bo)

        center = None
        bboxes = []
        centers = []
        # print('Detecting:--->')
        for num in range(0, len(bo)):
            if scos[num] > self.score_threshold and clas[0][num] == 3:
                count = count + 1
                (xmin, ymin, xmax, ymax) = (bo[num][1], bo[num][0], bo[num][3], bo[num][2])
                (left, right, top, bottom) = (xmin * im_width, xmax *\
                 im_width, ymin * im_height, ymax * im_height)
                new_top = top-((bottom-top) * self.extend_ratio_h)
                new_bottom = bottom + ((bottom - top) * self.extend_ratio_h)
                new_left = left - ((right - left) * self.extend_ratio_w)
                new_right = right + ((right - left) * self.extend_ratio_w)

                if new_top < 0:
                    new_top = 0
                if new_bottom > im_height:
                    new_bottom = im_height
                if new_left < 0:
                    new_left = 0
                if new_right > im_width:
                    new_right = im_width

                face_crop = image_np[int(top):int(bottom), int(left):int(right)]
                face_crop = cvtColor(face_crop, COLOR_BGR2RGB)
                center = (int(((new_right - new_left) / 2) + new_left), int(((new_bottom - new_top) / 2) + new_top))
                bboxes.append((left, right, top, bottom))
                centers.append(center)
                face_crop_list.append(face_crop)
            # break
        logging.info('Object detection completed')
        if center is None:
            return None, None, None
        else:
            #return (left, right, top, bottom), center, face_crop
            return bboxes, centers, face_crop_list
