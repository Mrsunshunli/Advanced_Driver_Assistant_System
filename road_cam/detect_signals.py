import cv2
import numpy as np
import time
from imutils.perspective import four_point_transform
#from imutils import contours
#import imutils

camera = cv2.VideoCapture(0)
counter = 0

def findTrafficSign():
    # define range HSV for blue color of the traffic sign
    lower_blue = np.array([85,100,70])
    upper_blue = np.array([115,255,255])

    while True:

        (grabbed, frame) = camera.read()

        if not grabbed:
            print("No input image")
            break
        
        frameArea = frame.shape[0]*frame.shape[1]
        
        # convert color image to HSV color scheme
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((3,3),np.uint8)
        
        # extract binary image with active blue regions
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        detectedTrafficSign = None
        largestArea = 0
        largestRect = None
        
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            for cnt in cnts:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # count euclidian distance for each side of the rectangle
                sideOne = np.linalg.norm(box[0]-box[1])
                sideTwo = np.linalg.norm(box[0]-box[3])
                # calculate area of the rectangle
                area = sideOne*sideTwo
                
                if area > largestArea:
                    largestArea = area
                    largestRect = box
            

        # draw contour of the found rectangle on  the original image
        if largestArea > frameArea*0.02:
            cv2.drawContours(frame,[largestRect],0,(0,0,255),2)
            

        #if largestRect is not None:
            # cut and warp interesting area
            warped = four_point_transform(mask, [largestRect][0])
            
            
            # use function to detect the sign on the found rectangle
            detectedTrafficSign = identifyTrafficSign(warped)
            #print(detectedTrafficSign)


            # write the description of the sign
            cv2.putText(frame, detectedTrafficSign, tuple(largestRect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        
        cv2.imshow("Signal recognition", frame)
        
        if cv2.waitKey(1) & 0xFF is ord('q'):
            cv2.destroyAllWindows()
            print("Stop programm and close all windows")
            break

def identifyTrafficSign(image):
    SIGNS_LOOKUP = {
        (1, 0, 0, 1): 'Turn Right', # turnRight
        (0, 0, 1, 1): 'Turn Left', # turnLeft
        (0, 1, 0, 1): 'Move Straight', # moveStraight
        (1, 0, 1, 1): 'Turn Back', # turnBack
        (0, 0, 0, 0): 'Stop', # Stop
    }

    THRESHOLD = 150
    
    image = cv2.bitwise_not(image)
    (subHeight, subWidth) = np.divide(image.shape, 10)
    subHeight = int(subHeight)
    subWidth = int(subWidth)

    # mark the ROIs borders on the image
    cv2.rectangle(image, (subWidth, 4*subHeight), (3*subWidth, 9*subHeight), (0,255,0),2) # left block
    cv2.rectangle(image, (4*subWidth, 4*subHeight), (6*subWidth, 9*subHeight), (0,255,0),2) # center block
    cv2.rectangle(image, (7*subWidth, 4*subHeight), (9*subWidth, 9*subHeight), (0,255,0),2) # right block
    cv2.rectangle(image, (3*subWidth, 2*subHeight), (7*subWidth, 4*subHeight), (0,255,0),2) # top block

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
    global counter
    if segments in SIGNS_LOOKUP:
        counter = counter + 1
        if(counter == 20):
            #print(SIGNS_LOOKUP[segments])
            if(SIGNS_LOOKUP[segments] == 'Turn Right'):
                s = "Right turn ahead"
            elif(SIGNS_LOOKUP[segments] == 'Turn Left'):
                s = "Left turn ahead"
            elif(SIGNS_LOOKUP[segments] == 'Move Straight'):
                s = "Keep moving straight"
            elif(SIGNS_LOOKUP[segments] == 'Turn Back'):
                s = "Slow down. U turn ahead"
            elif(SIGNS_LOOKUP[segments] == 'Stop'):
                s = "Stop!"
            print(s)
            counter = 0
        
        return SIGNS_LOOKUP[segments]
    else:
        return None


def main():
    findTrafficSign()


if __name__ == '__main__':
    main()
