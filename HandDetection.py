'''
Created by Daniel-Iosif Trubacs for the UOS AI SOCIETY on 2 January 2022.
A modules used to simplify the Hand Detection algorithm. To use just import the

'''

#importing the necessary libraries
from hashlib import new
import mediapipe as mp
import cv2 as cv
import numpy as np
import copy


# Setting the hand detection algorithm up (using mediapipe)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False)
mpDraw = mp.solutions.drawing_utils


class HandDetection:
    def __init__(self,img):
        self.img = img
        self.original_img = copy.deepcopy(img)

        # dimensions of the image
        self.height = int(self.img.shape[0])
        self.width = int(self.img.shape[1])

        # scaling up the image found
        self.alpha = int(0.05*self.height)
        self.beta = int(0.05*self.width)

    def highlight(self,show_hand,show_landmarks):

        #finding the landmarks in the image
        results = hands.process(self.img)

        # going through all the landmarks found
        parameters = results.multi_hand_landmarks
        if parameters:
                # the coordinates of all the landmarks found, used to find the rectangle containg the hand
                x_pos = []
                y_pos = []
                for handLms in parameters:
                    # appending all the coordinates of each landmark
                    for lm in handLms.landmark:
                        x_pos.append(lm.x)
                        y_pos.append(lm.y)
                    # showing the landmarks found in the image
                    if show_landmarks:
                        mpDraw.draw_landmarks(self.img, handLms, mpHands.HAND_CONNECTIONS)

                # the points defining the rectangle that contains the hand
                P_max = [int(self.width * min(x_pos)), int(self.height * min(y_pos))]
                P_min = [int(self.width * max(x_pos)), int(self.height * max(y_pos))]

                # a adjustesd (larger) rectangle for the hand
                P_max_0 = [int(self.width * min(x_pos)) - self.alpha, int(self.height * min(y_pos)) - self.beta]
                P_min_0 = [int(self.width * max(x_pos)) + self.alpha, int(self.height * max(y_pos)) + self.beta]

                # Get the size of the rectangle
                square_width = abs(P_max_0[0] - P_min_0[0])
                square_height = abs(P_max_0[1] - P_min_0[1])

                # Create the base of the new image
                new_image = np.zeros((max(square_width, square_height), max(square_width, square_height), 3), dtype=np.uint8)

                # Calculate the difference between the width and height
                diff = abs(square_width - square_height)
                
                # if the found rectangle's width is more than the height
                # create padding, calculate how much can be added more from the image
                # and add padding for the remaining
                if square_height < square_width:
                    top_padding = diff // 2
                    bottom_padding = diff - top_padding

                    new_y_min = max(P_max_0[1] - top_padding, 0)
                    new_y_max = min(P_min_0[1] + bottom_padding, self.height)
                    new_height = abs(new_y_max - new_y_min)
                    new_image[abs(square_height + diff - new_height)//2:abs(square_height + diff - new_height)//2+new_height, :, :] = self.original_img[new_y_min:new_y_max, P_max_0[0]:P_min_0[0], :]
                
                elif square_width < square_height:
                    left_padding = diff // 2
                    right_padding = diff - left_padding

                    new_x_min = max(P_max_0[0] - left_padding, 0)
                    new_x_max = min(P_min_0[0] + right_padding, self.width)
                    new_width = abs(new_x_max - new_x_min)
                    new_image[:, abs(square_width + diff - new_width)//2:abs(square_width + diff - new_width)//2+new_width, :] = self.original_img[P_max_0[1]:P_min_0[1], new_x_min:new_x_max, :]

                else:
                    new_image = self.original_img[P_max_0[1]:P_min_0[1], P_max_0[0]:P_min_0[0], :]


                cv.imshow("img", new_image)                
                    
                # highlighting the region containing the hand
                if show_hand:
                 cv.rectangle(self.img, P_min_0, P_max_0, (0, 0, 255), 2)


        return new_image




# trial
img = cv.imread('hand0.jpeg')
#img = cv.resize(img,[800,800])

Hand  = HandDetection(img)
img = Hand.highlight(show_hand=True, show_landmarks=True)



# cv.imshow("cool",img)
cv.waitKey()

