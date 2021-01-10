import cv2

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
faceClassif = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
cv2.namedWindow('Frame')
cv2.createTrackbar('Blur', 'Frame', 0, 15, nothing)
cv2.createTrackbar('Gray', 'Frame', 0, 1, nothing)

while True:
    ret, frame = cap.read()
    val = cv2.getTrackbarPos('Blur', 'Frame')
    grayVal = cv2.getTrackbarPos('Gray', 'Frame')

    if grayVal == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceClassif.detectMultiScale(
        frame, 
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30,30),
        maxSize=(200,200)
    )

    for (x,y,w,h) in faces:
        if val > 0: 
            frame[y:y+h, x:x+w] = cv2.blur(frame[y:y+h, x:x+w], (val,val))
    
    cv2.imshow('Frame', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()