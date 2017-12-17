import cv2

car_cascade = cv2.CascadeClassifier('cars.xml')
vid = cv2.VideoCapture('data/video1.avi')

while True:
    ret, img = vid.read()
    if (type(img) == type(None)):
        break
    
    # convert frames to gray scale for easier processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    # add bounding boxes
    for (x,y,w,h) in cars:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    
    cv2.imshow('video', img)

    if cv2.waitKey(33) == 27:
        break
cv2.destroyAllWindows()
