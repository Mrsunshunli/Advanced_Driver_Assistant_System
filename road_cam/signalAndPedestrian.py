import cv2
import numpy as np
import time
import imutils
from imutils.perspective import four_point_transform
from imutils.object_detection import non_max_suppression
from imutils import paths
import vlc

from playsound import playsound

#from imutils import contours
#import imutils

camera = cv2.VideoCapture(0)
counter1 = 0

person_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
counter = 0


def findTrafficSign():
    # define range HSV for blue color of the traffic sign
    lower_blue = np.array([85,100,70])
    upper_blue = np.array([115,255,255])

    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()

        if not grabbed:
            print("No input image")
            break
        
        #human detection
        image = frame.copy()
        image = imutils.resize(image, width=min(600, image.shape[1]))
        orig = image.copy()
 
        gray_frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rects = person_cascade.detectMultiScale(gray_frame)

        global counter
        if type(rects) == np.ndarray:
            counter += 1

        if(counter == 15):
            print("Human Detected")
            player = vlc.MediaPlayer('ped_det.mp3')
            player.play()
            #time.sleep(10)
            counter = 0
 
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
 
        # show the output images
        cv2.imshow("HUMAN DETECTION WINDOW", image)

        #frame = imutils.resize(frame, width=500)
        frameArea = frame.shape[0]*frame.shape[1]
        
        # convert color image to HSV color scheme
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # define kernel for smoothing   
        kernel = np.ones((3,3),np.uint8)
        # extract binary image with active blue regions
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # find contours in the mask
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        # defite string variable to hold detected sign description
        detectedTrafficSign = None
        
        # define variables to hold values during loop
        largestArea = 0
        largestRect = None
        
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            for cnt in cnts:
                # Rotated Rectangle. Here, bounding rectangle is drawn with minimum area,
                # so it considers the rotation also. The function used is cv2.minAreaRect().
                # It returns a Box2D structure which contains following detals -
                # ( center (x,y), (width, height), angle of rotation ).
                # But to draw this rectangle, we need 4 corners of the rectangle.
                # It is obtained by the function cv2.boxPoints()
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # count euclidian distance for each side of the rectangle
                sideOne = np.linalg.norm(box[0]-box[1])
                sideTwo = np.linalg.norm(box[0]-box[3])
                # count area of the rectangle
                area = sideOne*sideTwo
                # find the largest rectangle within all contours
                if area > largestArea:
                    largestArea = area
                    largestRect = box
            

        # draw contour of the found rectangle on  the original image
        if largestArea > frameArea*0.02:
            cv2.drawContours(frame,[largestRect],0,(0,0,255),2)
            


        #if largestRect is not None:
            # cut and warp interesting area
            warped = four_point_transform(mask, [largestRect][0])
            
            # show an image if rectangle was found
            #cv2.imshow("Warped", cv2.bitwise_not(warped))
            
            # use function to detect the sign on the found rectangle
            detectedTrafficSign = identifyTrafficSign(warped)
            #print(detectedTrafficSign)


            # write the description of the sign on the original image
            cv2.putText(frame, detectedTrafficSign, tuple(largestRect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        
        # show original image
        cv2.imshow("SIGN RECOGNITION WINDOW", frame)
        
        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF is ord('q'):
            cv2.destroyAllWindows()
            print("Stop programm and close all windows")
            break

def identifyTrafficSign(image):

    # define the dictionary of signs segments so we can identify
    # each signs on the image
    SIGNS_LOOKUP = {
        (1, 0, 0, 1): 'Turn Right', # turnRight
        (0, 0, 1, 1): 'Turn Left', # turnLeft
        (0, 1, 0, 1): 'Move Straight', # moveStraight
        (1, 0, 1, 1): 'Turn Back', # turnBack
    }

    THRESHOLD = 150
    
    image = cv2.bitwise_not(image)
    # (roiH, roiW) = roi.shape
    #subHeight = thresh.shape[0]/10
    #subWidth = thresh.shape[1]/10
    (subHeight, subWidth) = np.divide(image.shape, 10)
    subHeight = int(subHeight)
    subWidth = int(subWidth)

    # mark the ROIs borders on the image
    cv2.rectangle(image, (subWidth, 4*subHeight), (3*subWidth, 9*subHeight), (0,255,0),2) # left block
    cv2.rectangle(image, (4*subWidth, 4*subHeight), (6*subWidth, 9*subHeight), (0,255,0),2) # center block
    cv2.rectangle(image, (7*subWidth, 4*subHeight), (9*subWidth, 9*subHeight), (0,255,0),2) # right block
    cv2.rectangle(image, (3*subWidth, 2*subHeight), (7*subWidth, 4*subHeight), (0,255,0),2) # top block

    # substract 4 ROI of the sign thresh image
    leftBlock = image[4*subHeight:9*subHeight, subWidth:3*subWidth]
    centerBlock = image[4*subHeight:9*subHeight, 4*subWidth:6*subWidth]
    rightBlock = image[4*subHeight:9*subHeight, 7*subWidth:9*subWidth]
    topBlock = image[2*subHeight:4*subHeight, 3*subWidth:7*subWidth]

    # we now track the fraction of each ROI
    leftFraction = np.sum(leftBlock)/(leftBlock.shape[0]*leftBlock.shape[1])
    centerFraction = np.sum(centerBlock)/(centerBlock.shape[0]*centerBlock.shape[1])
    rightFraction = np.sum(rightBlock)/(rightBlock.shape[0]*rightBlock.shape[1])
    topFraction = np.sum(topBlock)/(topBlock.shape[0]*topBlock.shape[1])

    segments = (leftFraction, centerFraction, rightFraction, topFraction)
    segments = tuple(1 if segment > THRESHOLD else 0 for segment in segments)
    s = ""
    cv2.imshow("Warped", image)
    global counter1
    if segments in SIGNS_LOOKUP:
        counter1 = counter1 + 1
        if(counter1 == 10):
            #print(SIGNS_LOOKUP[segments])
            if(SIGNS_LOOKUP[segments] == 'Turn Right'):
                s = "Right turn ahead"
                player = vlc.MediaPlayer('rta.mp3')
                player.play()
            elif(SIGNS_LOOKUP[segments] == 'Turn Left'):
                s = "Left turn ahead"
                player = vlc.MediaPlayer('lta.mp3')
                player.play()
            elif(SIGNS_LOOKUP[segments] == 'Move Straight'):
                s = "Keep moving straight"
                #playsound("kms.mp3")
                player = vlc.MediaPlayer('kms.mp3')
                player.play()
            elif(SIGNS_LOOKUP[segments] == 'Turn Back'):
                s = "Slow down. U turn ahead"
                player = vlc.MediaPlayer('u_turn.mp3')
                player.play()
            print(s)
            counter1 = 0
        
        return SIGNS_LOOKUP[segments]
    else:
        return None


'''def human_detection():
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()

        if not grabbed:
            print("No input image")
            break
	
        image = frame.copy()
        image = imutils.resize(image, width=min(400, image.shape[1]))
        orig = image.copy()
 
        gray_frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rects = person_cascade.detectMultiScale(gray_frame)

        global counter
        if type(rects) == np.ndarray:
            counter += 1

        if(counter == 18):
            print("Human Detected")
            time.sleep(10)
            counter = 0
 
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
 
        # show the output images
        cv2.imshow("After NMS", image)


        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF is ord('q'):
            cv2.destroyAllWindows()
            print("Stop programm and close all windows")
            break'''


def main():
    findTrafficSign()


if __name__ == '__main__':
    main()
