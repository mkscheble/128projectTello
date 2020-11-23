#
# --- (c) 01/2020 f41ardu
#
# Tello cam using opencv proof of concept
# issue: huge delay -> Tipp scale down video size for improved performance on Raspberry PI 3+ 
# May also work with older versions of opencv supporting incomming udp streams. 
#
 
import numpy as np
# import opencv
import cv2
import time

import PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

telloVideo = cv2.VideoCapture("udp://@0.0.0.0:11111")
#telloVideo.set(cv2.CAP_PROP_FPS, 3)

# wait for frame
ret = False
# scale down 
scale = 3

#this is where I added//////////////////////////
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

fig = plt.figure()
nx = 4
ny = 3
for i in range(1, nx*ny+1):
    ax = fig.add_subplot(ny,nx, i)
    img = aruco.drawMarker(aruco_dict,i, 700)
    plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
    ax.axis("off")

# plt.show()

#this is where I added//////////////


while(True):
    # Capture frame-by-framestreamon
    ret, frame = telloVideo.read()
    if(ret): 
    # Our operations on the frame come here
        height , width , layers =  frame.shape
        new_h=int(height/scale)
        new_w=int(width/scale)
        resize = cv2.resize(frame, (new_w, new_h)) # <- resize for improved performance 
        # Display the resulting frame
        cv2.imshow('Tello',resize)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #this is where I added/////////////////
    frame = resize


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    plt.figure()
    plt.imshow(frame_markers, origin = "upper")
    if ids is not None:
        for i in range(len(ids)):
            c = corners[i][0]
            plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "+", label = "id={0}".format(ids[i]))
    """for points in rejectedImgPoints:
        y = points[:, 0]
        x = points[:, 1]
        plt.plot(x, y, ".m-", linewidth = 1.)"""
    plt.legend()
    plt.show()
    #this is where I added///////////////////

# When everything done, release the capture
telloVideo.release()
cv2.destroyAllWindows()

