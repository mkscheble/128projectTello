from djitellopy import Tello
import cv2
import numpy as np
import socket
import threading
import time
from time import sleep
import sys
import numpy as np
from queue import Queue
from queue import LifoQueue
 
 
######################################################################
width = 640  # WIDTH OF THE IMAGE
height = 480  # HEIGHT OF THE IMAGE
deadZone =100
######################################################################
 
startCounter = 0
 
# CONNECT TO TELLO
me = Tello()
me.connect()
me.for_back_velocity = 0
me.left_right_velocity = 0
me.up_down_velocity = 0
me.yaw_velocity = 0
me.speed = 0
 
 
 
print(me.get_battery())
sleep(10)
me.streamoff()
me.streamon()
########################
 
frameWidth = width
frameHeight = height
# cap = cv2.VideoCapture(1)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)
# cap.set(10,200)
 
State_data_file_name = 'statedata.txt'
index = 0
reference = 0.0     # Reference signal
control_LR = 0      # Control input for left/right
control_FB = 0      # Cotnrol input for forward/back
control_UD = 0      # Control input for up/down
control_YA = 0      # Control input for yaw
INTERVAL = 0.05  # update rate for state information
start_time = time.time()
dataQ = Queue()
stateQ = LifoQueue() # have top element available for reading present state by control loop

global imgContour
global dir;
def empty(a):
    pass
 
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV",640,240)
cv2.createTrackbar("HUE Min","HSV",20,179,empty)
cv2.createTrackbar("HUE Max","HSV",40,179,empty)
cv2.createTrackbar("SAT Min","HSV",148,255,empty)
cv2.createTrackbar("SAT Max","HSV",255,255,empty)
cv2.createTrackbar("VALUE Min","HSV",89,255,empty)
cv2.createTrackbar("VALUE Max","HSV",255,255,empty)
 
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",166,255,empty)
cv2.createTrackbar("Threshold2","Parameters",171,255,empty)
cv2.createTrackbar("Area","Parameters",1750,30000,empty)
 
def writeFileHeader(dataFileName):
    fileout = open(dataFileName,'w')
    #write out parameters in format which can be imported to Excel
    today = time.localtime()
    date = str(today.tm_year)+'/'+str(today.tm_mon)+'/'+str(today.tm_mday)+'  '
    date = date + str(today.tm_hour) +':' + str(today.tm_min)+':'+str(today.tm_sec)
    fileout.write('"Data file recorded ' + date + '"\n')
    # header information
    fileout.write('  index,   time,    ref,ctrl_LR,ctrl_FB,ctrl_UD,ctrl_YA,  pitch,   roll,    yaw,    vgx,    vgy,    vgz,   templ,   temph,    tof,      h,    bat,   baro,   time,    agx,    agy,    agz\n\r')
    fileout.close()


def writeDataFile(dataFileName):
    fileout = open(State_data_file_name, 'a')  # append
    print('writing data to file')
    while not dataQ.empty():
        telemdata = dataQ.get()
        np.savetxt(fileout , [telemdata], fmt='%7.3f', delimiter = ',')  # need to make telemdata a list
    fileout.close()



def report(str,index):
    telemdata=[]
    telemdata.append(index)
    telemdata.append(time.time()-start_time)
    telemdata.append(reference)
    telemdata.append(control_LR)
    telemdata.append(control_FB)
    telemdata.append(control_UD)
    telemdata.append(control_YA)
    data = str.split(';')
    data.pop() # get rid of last element, which is \\r\\n
    for value in data:
        temp = value.split(':')
        if temp[0] == 'mpry': # roll/pitch/yaw
            temp1 = temp[1].split(',')
            telemdata.append(float(temp1[0]))     # roll
            telemdata.append(float(temp1[1]))     # pitch
            telemdata.append(float(temp1[2]))     # yaw
            continue
        quantity = float(value.split(':')[1])
        telemdata.append(quantity)
    dataQ.put(telemdata)
    stateQ.put(telemdata)
    if (index %100) == 0:
        print(index, end=',')

def rcvstate():
    print('Started rcvstate thread')
    index = 0
    while not stateStop.is_set():
        
        response, ip = StateSock.recvfrom(1024)
        if response == 'ok':
            continue
        report(str(response),index)
        sleep(INTERVAL)
        index +=1
    print('finished rcvstate thread')

# Receive the message from Tello
def receive():
  # Continuously loop and listen for incoming messages
  while True:
    # Try to receive the message otherwise print the exception
    try:
      response, ip_address = CmdSock.recvfrom(128)
      print("Received message: " + response.decode(encoding='utf-8'))
    except Exception as e:
      # If there's an error close the socket and break out of the loop
      CmdSock.close()
      print("Error receiving: " + str(e))
      break


State_data_file_name = 'statedata.txt'

# receiveThread = threading.Thread(target=receive)
# receiveThread.daemon = True
# receiveThread.start()

# writeFileHeader(State_data_file_name)  # make sure file is created first so don't delay
# stateThread = threading.Thread(target=rcvstate)
# writeFileHeader(State_data_file_name1)
# stateThread.daemon = False  # want clean file close
# stateStop = threading.Event()
# stateStop.clear()
# stateThread.start()
file = open("statedata.txt", "w")
# file1 = open("statedatapos.txt", "w")
   

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
 
def getContours(img,imgContour):
    global dir
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # areaMin = cv2.getTrackbarPos("Area", "Parameters")
        areaMin = 2148
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            #print(len(approx))
            x , y , w, h = cv2.boundingRect(approx)
            cx = int(x + (w / 2))  # CENTER X OF THE OBJECT
            cy = int(y + (h / 2))  # CENTER X OF THE OBJECT
 
            if (cx <int(frameWidth/2)-deadZone):
                cv2.putText(imgContour, " GO LEFT " , (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
                cv2.rectangle(imgContour,(0,int(frameHeight/2-deadZone)),(int(frameWidth/2)-deadZone,int(frameHeight/2)+deadZone),(0,0,255),cv2.FILLED)
                dir = 1
            elif (cx > int(frameWidth / 2) + deadZone):
                cv2.putText(imgContour, " GO RIGHT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
                cv2.rectangle(imgContour,(int(frameWidth/2+deadZone),int(frameHeight/2-deadZone)),(frameWidth,int(frameHeight/2)+deadZone),(0,0,255),cv2.FILLED)
                dir = 2
            elif (cy < int(frameHeight / 2) - deadZone):
                cv2.putText(imgContour, " GO UP ", (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
                cv2.rectangle(imgContour,(int(frameWidth/2-deadZone),0),(int(frameWidth/2+deadZone),int(frameHeight/2)-deadZone),(0,0,255),cv2.FILLED)
                dir = 3
            elif (cy > int(frameHeight / 2) + deadZone):
                cv2.putText(imgContour, " GO DOWN ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255), 3)
                cv2.rectangle(imgContour,(int(frameWidth/2-deadZone),int(frameHeight/2)+deadZone),(int(frameWidth/2+deadZone),frameHeight),(0,0,255),cv2.FILLED)
                dir = 4
            else: dir=0
 
            cv2.line(imgContour, (int(frameWidth/2),int(frameHeight/2)), (cx,cy),(0, 0, 255), 3)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,(0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,(0, 255, 0), 2)
            cv2.putText(imgContour, " " + str(int(x)) + " " + str(int(y)), (x - 20, y - 45), cv2.FONT_HERSHEY_COMPLEX,0.7,(0, 255, 0), 2)
        else: dir=0
 
def display(img):
    cv2.line(img,(int(frameWidth/2)-deadZone,0),(int(frameWidth/2)-deadZone,frameHeight),(255,255,0),3)
    cv2.line(img,(int(frameWidth/2)+deadZone,0),(int(frameWidth/2)+deadZone,frameHeight),(255,255,0),3)
    cv2.circle(img,(int(frameWidth/2),int(frameHeight/2)),5,(0,0,255),5)
    cv2.line(img, (0,int(frameHeight / 2) - deadZone), (frameWidth,int(frameHeight / 2) - deadZone), (255, 255, 0), 3)
    cv2.line(img, (0, int(frameHeight / 2) + deadZone), (frameWidth, int(frameHeight / 2) + deadZone), (255, 255, 0), 3)
 
while True:
 
    # GET THE IMAGE FROM TELLO
    frame_read = me.get_frame_read()
    myFrame = frame_read.frame
    img = cv2.resize(myFrame, (width, height))
    imgContour = img.copy()
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
    h_min = 0
    h_max = 179
    s_min = 0
    s_max = 255
    v_min = 75
    v_max = 194
    # h_min = cv2.getTrackbarPos("HUE Min","HSV")
    # h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    # s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    # s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    # v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    # v_max = cv2.getTrackbarPos("VALUE Max", "HSV")
 
 
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHsv,lower,upper)
    result = cv2.bitwise_and(img,img, mask = mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
 
    imgBlur = cv2.GaussianBlur(result, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    # threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    # threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    threshold1 = 221
    threshold2 = 248
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    getContours(imgDil, imgContour)
    display(imgContour)
 
    ################ FLIGHT
    if startCounter == 0:
       me.takeoff()
       startCounter = 1
       sleep(5)
 
    position = 0
    if dir == 1:
    #    me.yaw_velocity = -60
       me.move_left(20)
       position -= 20
       print(-601)
    elif dir == 2:
    #    me.yaw_velocity = 60
       me.move_right(20)
       position += 20
       print(602)
    elif dir == 3:
       me.up_down_velocity= 60
       print(603)
    elif dir == 4:
       me.up_down_velocity= -60
       print(-604)
    else:
       me.left_right_velocity = 0; me.for_back_velocity = 0;me.up_down_velocity = 0; me.yaw_velocity = 0
   # SEND VELOCITY VALUES TO TELLO
    if me.send_rc_control:
       me.send_rc_control(me.left_right_velocity, me.for_back_velocity, me.up_down_velocity, me.yaw_velocity)
    print(dir)
    print(me.get_battery())
    file.write("%s, %s" % (str(position), str(time.time())))
    file.flush()
    # file1.write(position)
    # file1.flush()
 
    stack = stackImages(0.9, ([img, result], [imgDil, imgContour]))
    cv2.imshow('Horizontal Stacking', stack)

    if cv2.waitKey(1) & 0xFF == ord('q') or input('') == 'quit':
        me.land()
        stateStop.set()  # set stop variable
        stateThread.join()   # wait for termination of state thread before closing socket
        # writeDataFile(State_data_file_name)
        file.close()
        # file1.close()
        break

    # except KeyboardInterrupt as e:
    #     me.land()
    #     stateStop.set()  # set stop variable
    #     stateThread.join()   # wait for termination of state thread before closing socket
    #     writeDataFile(State_data_file_name)
    # break
 
# cap.release()
cv2.destroyAllWindows()