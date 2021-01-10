import cv2

faceClassif = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')

def nothing(val):
    pass

image = cv2.imread('../oficina2.jpg')

cv2.namedWindow('Imagen')
cv2.createTrackbar('Blur', 'Imagen', 0, 15, nothing)
cv2.createTrackbar('Gray', 'Imagen', 0, 1, nothing)


while True:
    val = cv2.getTrackbarPos('Blur', 'Imagen')
    grayVal = cv2.getTrackbarPos('Gray', 'Imagen')

    if grayVal == 1:
        imageN = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    else: imageN = image.copy()

    faces = faceClassif.detectMultiScale(
        imageN, 
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        maxSize=(200,200)
    )

    for (x,y,w,h) in faces:
        if val > 0:
            imageN[y:y+h, x:x+w] = cv2.blur(imageN[y:y+h, x:x+w], (val,val))

    cv2.imshow('Imagen', imageN)
    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()