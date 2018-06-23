import cv2
if __name__ == "__main__":

    #test video
    path = "C:/Users/joyachen/Desktop/IMG_1246.mp4"
    cap = cv2.VideoCapture(path)
    while True:
        ret,frame = cap.read()
        cv2.imshow("window",frame)
        cv2.waitKey(1)
