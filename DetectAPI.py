"""
# Title: DetectAPI
# Application: DetectAPI to detect the objects(cars in this case.)
# Author: Nitesh Kumar M
# Date Created: 05/12/2018
# Description:  DetectAPI
"""
# pylint: disable=E0611
from __future__ import print_function
import logging
from cv2 import imread, imwrite
from detection import Detection

class DetectAPI():
    '''
    Description : Class ObjDetectAPI to detect the objects(cars in this case)
    '''
    def __init__(self):
        self.objdetect = Detection()

    def get_object(self, image):
        '''
        Description: Detect object from a given image
        Arguments: input image
        Returns: cropped object from the image
        '''
        logging.info('Detecting Objects')
        bboxes, centeres, objects = self.objdetect.detect_object(image)
        return bboxes, centeres, objects

if __name__ == "__main__":
    OBJECT_DETECTAPI = DetectAPI()
    IMAGE = imread('./car.jpg')
    BBOXES, CENTERS, OBJECTS = OBJECT_DETECTAPI.get_object(IMAGE)
    for car in OBJECTS:
      imwrite('car.jpg', car)
