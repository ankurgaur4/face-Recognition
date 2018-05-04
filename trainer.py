import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
    # get the path of all files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empty face list and id
    faceSamples=[]
    Ids = []

    for imagePath in imagePaths:
        # load the image and convert it to gray scale using Pillow
        pilImage = Image.open(imagePath).convert('L')
        #Now converting the pil image into numpy array
        imageNP = np.array(pilImage, 'uint8')
        #getting the id from image
        Id = int(os.path.split(imagePath)[-1].split(".")[0])
        #Extract the face from the training image sample
        faces = detector.detectMultiScale(imageNP)
        for (x,y,w,h) in faces:
            faceSamples.append(imageNP[y:y+h,x:x+w])
            Ids.append(Id)
            
    return faceSamples,Ids
    
faces,Ids = getImagesAndLabels('data')
recognizer.train(faces, np.array(Ids))
recognizer.save('trainner/trainner.yml')
