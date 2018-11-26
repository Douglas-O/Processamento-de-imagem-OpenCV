import cv2
import numpy as np
import time

# Capture the input frame from video
def get_frame(cap, scaling_factor):
    # Capture the frame from video capture object
    ret, frame= cap.read()

    # Resize the input frame
    frame = cv2.resize(frame, None, fx=scaling_factor,
            fy=scaling_factor, interpolation=cv2.INTER_AREA)

    return frame

if __name__=='__main__':
    cap = cv2.VideoCapture('videoplayback.mp4')
    scaling_factor = 0.8
    averaged_frame = cv2.imread('averaged_frame.jpg')
    averaged_frame = cv2.resize(averaged_frame, None, fx=scaling_factor,
            fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # Iterate until the user presses ESC key or video stops
    while cap.isOpened():
        #time.sleep(0.1)
        frame = get_frame(cap, scaling_factor)

        # Convert the HSV colorspace
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define BGR range in HSV colorspace
        #color = np.uint8([[[0,255,0 ]]])
        #hsv_color = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
        #print (hsv_color)

        #yellow
        lower = np.array([-40, 10, 10])
        upper = np.array([40, 255, 255])

        #red
        #lower = np.array([160, 10, 10])
        #upper = np.array([200, 255, 255])

        #blue
        #lower = np.array([100, 80, 80])
        #upper = np.array([140, 255, 255])

        #green
        #lower = np.array([40, 30, 30])
        #upper = np.array([80, 255, 255])

        #all
        #lower = np.array([-255, 1, 1])
        #upper = np.array([255, 255, 255])

        # Threshold the HSV image to get only BGR color
        mask = cv2.inRange(hsv, lower, upper)
        mask_inv = cv2.bitwise_not(mask)

        # Bitwise-AND mask and original image, averaged frame and inverted mask
        foreground = cv2.bitwise_and(frame, frame, mask=mask)
        background = cv2.bitwise_and(averaged_frame, averaged_frame, mask=mask_inv)
        res = cv2.add(background,foreground)
        res = cv2.medianBlur(res, 3)

        cv2.imshow('Original image', frame)
        cv2.imshow('Color Detector', res)
        cv2.imshow('Averaged frame', averaged_frame)

        # Check if the user pressed ESC key
        c = cv2.waitKey(5)
        if c == 27:
            break

    cv2.destroyAllWindows()