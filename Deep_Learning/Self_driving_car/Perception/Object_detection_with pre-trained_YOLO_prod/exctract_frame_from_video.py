import cv2

if __name__ == '__main__':
    # Opens the Video file
    cap= cv2.VideoCapture('road_video.mp4')
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite('images/img'+str(i)+'.jpg',frame)
        i+=1

    cap.release()
    cv2.destroyAllWindows()
