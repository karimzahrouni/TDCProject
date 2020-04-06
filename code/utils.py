import dlib
import numpy as np
import time
import cv2
import os
import face_recognition
import glob
import os,smtplib,imutils
import serial
SERIAL_PORT = "/dev/serial0"
running = True

'''
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
prototxt = '/home/zahrouni/testPy/TDC_RPI/MobileNetSSD_deploy.prototxt.txt' 
model = '/home/zahrouni/testPy/TDC_RPI/MobileNetSSD_deploy.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt,model)
coor=[]
MIN_PREDECT = 0.6 

def Objectclass(frame):
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
    net.setInput(blob)  # me blob eka nural network ekata deela eke detections gannawa
    detections = net.forward()  # forward method eken return wenne <type 'numpy.ndarray'>.
    for i in np.arange(0, detections.shape[2]):  
        confidence = detections[0, 0, i, 2]
        if confidence > MIN_PREDECT :
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            if label == 'person' :
                cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                return True
            else :
                return False
'''
cascPath = "haarcascade.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

def face_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv_flag = cv2.CASCADE_SCALE_IMAGE
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv_flag)
    for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def Alert(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv_flag = cv2.CASCADE_SCALE_IMAGE
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv_flag)
    for (x,y,w,h) in faces:
        print("personne detect")
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        path = ("image/image.jpg")
        cv2.imwrite(path,frame)
        path = ("image/image.jpg")
        sendemail(path)
        #s+=1

# In the NMEA message, the position gets transmitted as:
# DDMM.MMMMM, where DD denotes the degrees and MM.MMMMM denotes
# the minutes. However, I want to convert this format to the following:
# DD.MMMM. This method converts a transmitted string to the desired format
def formatDegreesMinutes(coordinates, digits):
    
    parts = coordinates.split(".")

    if (len(parts) != 2):
        return coordinates

    if (digits > 3 or digits < 2):
        return coordinates
    
    left = parts[0]
    right = parts[1]
    degrees = str(left[:digits])
    minutes = str(right[:3])

    return degrees + "." + minutes

# This method reads the data from the serial port, the GPS dongle is attached to,
# and then parses the NMEA messages it transmits.
# gps is the serial port, that's used to communicate with the GPS adapter
def getPositionData(gps):
    data = gps.readline()
    message = data[0:6]
    if (message == "$GPRMC"):
        # GPRMC = Recommended minimum specific GPS/Transit data
        # Reading the GPS fix data is an alternative approach that also works
        parts = data.split(",")
        if parts[2] == 'V':
            # V = Warning, most likely, there are no satellites in view...
            print("GPS receiver warning")
        else:
            # Get the position data that was transmitted with the GPRMC message
            # In this example, I'm only interested in the longitude and latitude
            # for other values, that can be read, refer to: http://aprs.gids.nl/nmea/#rmc
            longitude = formatDegreesMinutes(parts[5], 3)
            latitude = formatDegreesMinutes(parts[3], 2)
            print("Your position: lon = " + str(longitude) + ", lat = " + str(latitude))
            return longitude,latitude
    else:
        # Handle other NMEA messages and unsupported strings
        pass


def sendemail(ImgFileName) :
    #os.system('ssmtp xxxxxxxxx@gmail.com < text.txt')
    img_data = open(ImgFileName, 'rb').read()
    msg = MIMEMultipart()
    Server = "smtp.gmail.com"
    Port = 587
    gps = serial.Serial(SERIAL_PORT, baudrate = 9600, timeout = 0.5)
    lat , longt = getPositionData(gps)
    Subject = 'SecurityTemp'
    From = 'yyyyyyyyyy@gmail.com'
    To = 'xxxxxxxxx@gmail.com'
    text = MIMEText("zone position: lon = " + str(longt) + ", lat = " + str(lat))
    image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
    msg.attach(image)
    s = smtplib.SMTP(Server, Port)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login("xxxxxxx@gmail.com","**********")
    s.sendmail(From, To, msg.as_string())
    s.quit()
