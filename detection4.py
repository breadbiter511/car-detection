import cv2 #importing library
capture = cv2.VideoCapture("cardetection/7303988-hd_1920_1080_30fps.mp4")#capture frames from the video
mlmodel = cv2.CascadeClassifier("cardetection/cars.xml")#loading the machine learning model to detect veichles
while True:
    recieve,frames = capture.read()#this rewads all the individual frames in the video
    grey = cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)#converting into greyscale format
    multiplecars = mlmodel.detectMultiScale(grey,1.1,1)# it will detect the cars of different sizes
    for(x,y,w,h) in multiplecars:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(255,0,0),3)#this will make a rectangle around every car
    cv2.imshow("myvideo",frames)#it will display all the frames in the window
    if cv2.waitKey(33) == 27:#to stop the video we will have to press escape
        break
cv2.destroyAllWindows()