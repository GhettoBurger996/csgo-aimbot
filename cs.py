import time
import cv2
import mss
import numpy
import pyscreenshot as ImageGrab

start_time = time.time()
title = "[PIL.ImageGrab] FPS benchmark"

mon = (0, 40, 800, 640)
monitor = {"top": 40, "left": 0, "width": 800, "height": 640}

fps = 0
sct = mss.mss()
display_time = 2

'''def screen_recordPYScreen():
    global fps, start_time 
    
    while True:
        img = numpy.asarray(ImageGrab.grab(bbox=mon))
        cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fps += 1
        TIME = time.time() - start_time
        
        if (TIME) >= display_time:
            print("FPS: ", fps / TIME)
            fps = 0
            start_time = time.time()

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break'''

def screen_recordMSS():
    global fps, start_time

    while True:
        img = numpy.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fps+=1
        TIME = time.time() - start_time

        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

#screen_recordPYScreen()
screen_recordMSS()