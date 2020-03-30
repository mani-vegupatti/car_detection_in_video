# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:45:29 2020

@author: ajink / manivs
"""
from yolo import YOLO 
from PIL import Image
import cv2
import csv
import numpy as np
import argparse
from DetectColour import DetectColour
from CarObject import CarObject
from Frame import Frame
from DetectCar import DetectCar
# Begin Mani
from DetectCartype import DetectCartype
# End

#Begin Mani
video = "./video.mp4"
#End
#Create an object of detect car class
detect_car = DetectCar()
#Create an object of detect colour class
detect_colour =DetectColour()
#Start Mani
detect_cartype = DetectCartype()
#End

"""Black=(np.array([  0,   0,  27]), np.array([131,  65,  65]))
Silver=(np.array([117,   0, 122]), np.array([255,  19, 255]))
Red=(np.array([155,  56,  60]), np.array([190, 170, 197]))
White=(np.array([ 73,   0, 178]), np.array([140,  21, 255]))
Blue=(np.array([ 67,  23,  66]), np.array([116, 106, 255]))"""

result_row = {}
results=[]
def main():
    video_capture = cv2.VideoCapture(video)

    """out=args.out
    """
    out = "./out/"
    count=0
    frame_no = 1
    colour_types = {  'Black':0,
            'Silver':0,
            'Red':0,
            'White':0,
            'Blue':0
            }
   
    while True:
        result_row = {  "FrameNo":frame_no, 
                        "Sedan": colour_types.copy(), 
                        "Hatchback": colour_types.copy(), 
                        "Total":0
                        }            
        # Capture the input video frame by frame
        ret, frame = video_capture.read()
        if ret == True:
            #result_row[frame_no] = {"Sedan": colour_types.copy(), "Hatchback": colour_types.copy(), "Total":0}
            #result_row["FrameNo"] = frame_no
            frame_details = detect_car.detect_Car(frame_no, frame)
            #cars_count = 0
            for bounding_box in frame_details.getObjects():#For each object in given frame 
                #Get the hsv value of given image
                hsv=detect_colour.get_hsv(bounding_box)
                #Get the colour of object(Car) from image
                colour=detect_colour.get_colour(hsv)
                car_type = detect_cartype.predict_cartype(bounding_box)
                if car_type == "Hatchback":
                     #result_row[frame_no]["Hatchback"][colour] += 1
                     result_row["Hatchback"][colour] += 1
                else:
                    #result_row[frame_no]["Sedan"][colour] += 1
                    result_row["Sedan"][colour] += 1
                #print(car_type)
                #print(colour)#print colour
                #cv2.imshow("Bounding box", bounding_box)#Show image in new window
                #cv2.imwrite(out+str(count)+'.png',bounding_box) #Save image on out directory
                #print("<---- Frame Processed----->")
                count=count+1 #Increase the count
                #cars_count += 1
                #result_row[frame_no]["Total"] += 1
                result_row["Total"] += 1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
        #result_row[frame]["Total"] = cars_count  
        #if frame_no >= 50:
        #    break
        frame_no=frame_no+1
        #print("*******")
        #print(frame_details.getFrame())
        #print(frame_details.getObjects())
        #print("Count of Cars:" + str(cars_count))
        if frame_no >= 1490:
            print(result_row)
        results.append(result_row)
        #print("*******")
        
    #Once the video is over release video capture
    video_capture.release()
    # Closes all the windows
    cv2.destroyAllWindows()
    
    with open('predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for result_row in results:
            csv_row = []
            csv_row.append(result_row["FrameNo"])
            for colour_count in result_row["Sedan"].values():
                csv_row.append(colour_count)
            for colour_count in result_row["Hatchback"].values():
                csv_row.append(colour_count)
            csv_row.append(result_row["Total"])
            writer.writerow(csv_row)


if __name__ == "__main__":
    main()
 