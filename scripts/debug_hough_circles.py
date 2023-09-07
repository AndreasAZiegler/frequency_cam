import cv2 as cv
import numpy as np
import os

directory = "/data/ros_ws/calibration_ws/debug_frames/"

for filename in os.listdir(directory):
    file = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(file):
        image = cv.imread(file, cv.IMREAD_COLOR)

        # gray = np.uint8(image)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # gray = cv.medianBlur(gray, 5)

        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1.0, 10, param1=1, param2=4, minRadius=0, maxRadius=10);

        # print("Nur. of circles: " + str(len(circles)))

        if circles is not None:
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(image, center, 1, (0, 100, 100), 1)
                # circle outline
                radius = i[2]
                cv.circle(image, center, radius, (255, 0, 255), 1)

            if len(circles[0]) == 3:
                print("Image: " + str(file))
                cv.imshow("window", image)
                cv.waitKey(0)

            else:
                print("Image: " + str(file))
                print("Nr. of circles: " + str(len(circles[0])))
                cv.imshow("window", image)
                cv.waitKey(0)
        else:
            print("Image: " + str(file))
            print("No circles detected.")
            cv.imshow("window", image)
            cv.waitKey(0)
