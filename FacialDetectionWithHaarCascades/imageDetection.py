import cv2

faceClassif = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')

image = cv2.imread('../oficina.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

""" 
  El primer parametro recibira la imagen puede ser la imagen normal o en escala de grises
  (recomendado)
  El segundo valor (scaleFactor) va indicar que tanto va a ser reducida la 
  imagen en este caso se uso un 10% para un 30% se puede poner 1.3
  El tercer parametro (minNeighbors) indica cuantos vecinos debe tener cada rectangulo candidato
  para retener
  Los ultimos parametros minSize y maxSize se refiere al tamaño del cuadro que se puede detectar 
  en una imagen si es menor que el tamaño escrito entonces el algoritmo no lo detectara y lo
  mismo ocurrira con el tamaño maximo
"""
faces = faceClassif.detectMultiScale(
    gray, 
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30,30),
    maxSize=(200,200)
)

""" Dibuja el rectangulo """
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()