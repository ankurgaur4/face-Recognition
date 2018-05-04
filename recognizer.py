import pyttsx
import cv2
import numpy as np
engine = pyttsx.init()
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im, (x,y), (x+w,y+h), (225,0,0), 2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf>50):
            if(Id==1):
                Id = "Tara Chand"
                engine.say("Tara Chand")
                engine.runAndWait()
            elif(Id==2):
                Id = "Yogesh Jangid"
                engine.say("Yogesh Jangid")
                engine.runAndWait()
            elif(Id==3):
                Id = "Ankur Gaur"
                engine.say("Ankur Gaur")
                engine.runAndWait()
            elif(Id==4):
                Id = "Shivam"
                engine.say("Shivam")
                engine.runAndWait()
            elif(Id==5):
                Id = "Ashish Soni"
                engine.say("Ashish Soni")
                engine.runAndWait()
            elif(Id==9):
                Id = "Ankur Gaur"
                engine.say("Ankur Gaur")
                engine.runAndWait()
        else:
            Id = "Unknown"
            engine.say("Unknown")
            engine.runAndWait()
        cv2.putText(im, str(Id), (x+w,y+h), font,1, (0,255,0), 2, cv2.LINE_AA)  
    cv2.imshow("Image",im)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
                
