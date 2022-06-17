import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'path of images folder'
images = []
classNames = []
myList = os.listdir(path)
print(myList)



for cls in myList:
    curImg= cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

print(classNames)

def findencoiding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAtt(name):
    with open('path of csv file','r+') as f:
        myData = f.readlines()
        nameList = []
        print(myData)
        for line in myData:
            entry =line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now= datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



encodeListknown = findencoiding(images)
print('Encoding compleate')

cap = cv2.VideoCapture("cam number for inbuilt its 0")

while True:
    success , img = cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs =cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgs)
    encodesCurFrame = face_recognition.face_encodings(imgs,facesCurFrame)

    for encodeface, faceloc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListknown,encodeface)
        faceDis = face_recognition.face_distance(encodeListknown,encodeface)
        #print (faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)

            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 =y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAtt(name)

    if cv2.waitKey(10)== ord('q'):
        break
            
    cv2.imshow('webcam',img)
    cv2.waitKey(1)

    #github:- @padalakiran 
