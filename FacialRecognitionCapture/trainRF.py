import cv2
import os
import numpy as np

dataPath = './data'
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo las imagenes')

    for filename in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + filename)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+ filename, 0))
        image = cv2.imread(personPath+'/'+filename,0)
        #cv2.imshow('image', image)
        #cv2.waitKey(10)
    
    label += 1

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenando el reconocedor de rostros
print('Entrenando...')
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido
face_recognizer.write('modeloLBPHFace.xml')
print('Modelo almacenado...')