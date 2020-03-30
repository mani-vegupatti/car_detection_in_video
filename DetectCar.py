# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:45:29 2020

@author: ajink / manivs
"""
from yolo import YOLO 
from PIL import Image
import cv2
import numpy as np
import argparse
from CarObject import CarObject
from Frame import Frame

class DetectCar():
    def __init__(self):
        self.yolo = YOLO()

    def detect_Car(self, frame_no, frame):
        ObjectsInFrame=[]
        image = Image.fromarray(frame) # Get each frame of video
        
        r_image, object_boxes, time_value = self.yolo.detect_image(image) #Detect objects from frame and retun image and coordinates with class
        result = np.asarray(r_image)#Save resulting image 
        #print('Size of box {}'.format((object_boxes))) #Print Box size
        
        #cv2.imshow("Yolo Test", result) #Show the result image with detected object.
        # Use Q to quit from the window
        bounding_boxes = []
        for object_box in object_boxes:#For each object in given frame 
            if object_box['class'] == 'car': #Select object with only car as class
                #print('Found a car in frame')
                top = object_box['top']-20
                left = object_box['left']
                bottom = object_box['bottom']
                right = object_box['right']
                bounding_box = frame[top:bottom, left:right] # Resulting image of car
                
                bounding_boxes.append(bounding_box)

                #Car=CarObject(bounding_box,'car')
                #ObjectsInFrame.append(Car)
            
        frame_details=Frame(frame_no,bounding_boxes)
        return frame_details

