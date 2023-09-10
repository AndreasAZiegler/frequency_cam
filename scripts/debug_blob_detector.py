import cv2 as cv
import numpy as np
import os

directory = "/data/ros_ws/calibration_ws/debug_frames/"

for filename in os.listdir(directory):
    file = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(file):
        # image = cv.imread(file, cv.IMREAD_COLOR)
        image = cv.imread(file, cv.IMREAD_GRAYSCALE)

        # gray = np.uint8(image)
        # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # gray = cv.medianBlur(gray, 5)

        # Setup SimpleBlobDetector parameters.
        params = cv.SimpleBlobDetector_Params()
         
        # Change thresholds
        params.minThreshold = 200;
        params.maxThreshold = 260;
         
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 20
         
        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.7
         
        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.5
         
        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.5

        params.minDistBetweenBlobs = 10

        params.blobColor = 255

        # Set up the detector with default parameters.
        # detector = cv.SimpleBlobDetector()
        detector = cv.SimpleBlobDetector_create(params)
         
        # Detect blobs.
        keypoints = detector.detect(image)
         
        # print("Nur. of circles: " + str(len(circles)))

        if len(keypoints) != 0:
            # Draw detected blobs as red circles.
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
            # image = cv.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            for keypoint in keypoints:
                cv.circle(image, (int(keypoint.pt[0]), int(keypoint.pt[1])), 1, (0, 100, 100), 1)

            # if len(circles[0]) == 3:
            print("Image: " + str(file))
            print("len(keypoints): " + str(len(keypoints)))
            cv.imshow("window", image)
            cv.waitKey(0)

            # else:
            #     print("Image: " + str(file))
            #     print("Nr. of circles: " + str(len(circles[0])))
            #     cv.imshow("window", image)
            #     cv.waitKey(0)

        else:
            print("Image: " + str(file))
            print("No circles detected.")
            cv.imshow("window", image)
            cv.waitKey(0)
