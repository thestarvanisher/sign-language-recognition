#importing the necessary libraries
import cv2 as cv
import numpy as np
import time

#DEFINIG GLOBAL VARIABLES AND FUNCTIONS USED IN ALL THE CLASSES
#a kernel for dilation and erosion
kernel_noise = np.ones((3,3), np.uint8)

#kernel for closing the picture
kernel_closing = np.ones((7,7), np.uint8)

#chosing the font
font = cv.FONT_HERSHEY_SIMPLEX

# a fucntion to work out the area between 2 points (in xOy coordinates)
def area (P_1,P_2):
    return abs((P_2[0]-P_1[0])*(P_2[0]-P_1[0]))


def img_edge(img):
        #channig the image to gray
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        #bluring the image 
        img = cv.blur(img,(5,5))
        
        #eroding the image to remove the noise
        #img = cv.erode(img,kernel_noise,iterations=10)
        
        #dilating the image to pronounce the edges
        #img = cv.dilate(img, kernel_noise, iterations=10)
        
    
        #optimal threshold (used for better img det)
        img  = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv.THRESH_BINARY, 15, 0)
        
        
        #eroding the image to remove the noise
        img = cv.erode(img,kernel_noise,iterations=1)
        
        #closing the image
        #img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_closing)
        
        #dilating the image to pronounce the edges
        img = cv.dilate(img, kernel_noise, iterations=1)
        
        #adding a median blur to remove the unwanted noise
        img = cv.medianBlur(img,9)
        
        #eroding the image to remove the noise
        img = cv.erode(img,kernel_noise,iterations=1)
        
        #closing the image
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_closing)
        
        #dilating the image to pronounce the edges
        img = cv.dilate(img, kernel_noise, iterations=1)
            
        #finding the edges in the image
        img = cv.Canny(img,50,50)
        
        #adding a median blur to remove the unwanted noise
        img = cv.medianBlur(img,1)
        
        #dilating the image to pronounce the edges
        img = cv.dilate(img, kernel_noise, iterations=1)
            
        return img

# finding the object from a contours list. n_ob the number of objects u want to find
# width and height of the original image required for some imgae manipulation
def find_objects(contours,n_ob,width,height):
    
   # the array of objects found
   objects = []

   # going through all the contours found
   for i in range(len(contours)):
     # reshaping the contours in a list of 2d points
     shape = contours[i].shape
     aux = np.reshape(contours[i],(shape[0],shape[2]))
    
     #finding the 2 points that define the rectangle 
     P_min = (max(aux[:,0]),max(aux[:,1]))
     P_max = (min(aux[:,0]),min(aux[:,1]))
     
     #the area inside the contour
     cont_area  = area(P_min,P_max)
     
     # the objects found
     if cont_area < 0.95*width*height and cont_area > 1:
      objects.append((P_min,P_max,cont_area))
   
   #sorting the array of objects
   sorted_objects = sorted(objects,key=lambda x:x[2]) 
   
   #returning the points that define the rectangles (objects)
   if len(sorted_objects)>n_ob:
    return sorted_objects[len(sorted_objects)-n_ob:]
   else:
    return sorted_objects
   

class img_manipulation:
    def __init__(self):
        pass
    
    # img is the image that has to be rescaled and f is the fraction by which
    # the iamge will be rescaled
    def rescale(img,f):
         # getting the new dimensions
         height = int(f*img.shape[0])
         width = int(f*img.shape[1])
         dim = (width,height)
         
         # resizing the image
         img = cv.resize(img,dim,interpolation=cv.INTER_AREA)
         
         #returning the rescaled image
         return img
    
    # a normal resize function /  made only to not use all the cv functions every time 
    def resize(img,height,width):
        # getting the dimensions
        dim = (int(width),int(height))
        
        # resizing the image
        img = cv.resize(img,dim,interpolation=cv.INTER_AREA)
        
        #returning the resized image
        return img
    
    # a function that transforms an image only in the edges
    def rgb_to_edges(img):
        
        #creating a copy of the original
        img_original = img.copy()
        
        
        
        #channig the image to gray
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        #bluring the image 
        img = cv.blur(img,(5,5))
        
        #eroding the image to remove the noise
        #img = cv.erode(img,kernel_noise,iterations=10)
        
        #dilating the image to pronounce the edges
        #img = cv.dilate(img, kernel_noise, iterations=10)
        
    
        #optimal threshold (used for better img det)
        img_t = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv.THRESH_BINARY, 15, 0)
        
        
        #eroding the image to remove the noise
        img = cv.erode(img_t,kernel_noise,iterations=1)
        
        #closing the image
        #img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_closing)
        
        #dilating the image to pronounce the edges
        img = cv.dilate(img, kernel_noise, iterations=1)
        
        #adding a median blur to remove the unwanted noise
        img = cv.medianBlur(img,9)
        
        #eroding the image to remove the noise
        img = cv.erode(img,kernel_noise,iterations=1)
        
        #closing the image
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_closing)
        
        #dilating the image to pronounce the edges
        img = cv.dilate(img, kernel_noise, iterations=1)
            
        #finding the edges in the image
        img = cv.Canny(img,50,50)
        
        #adding a median blur to remove the unwanted noise
        img = cv.medianBlur(img,1)
        
        #dilating the image to pronounce the edges
        img = cv.dilate(img, kernel_noise, iterations=1)
        
        #closing the image
        #img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_closing)
        
        
        #showing the contours
        contours, hierarchy = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        cv.drawContours(img_original, contours, -1, (0,0,255), 1)
            
        return img_t



class obj_det:
    def __init__(self):
        pass
    
    
    # a function that finds the object in a image 
    # n_ob the maximum number of objects we are looking foor
    def find_obj(img,n_ob):
        #measuring the time 
        start_1 = time.time()
        #creating a copy of the orignal image
        img_original = img.copy()  
        
        # finding the edges in the image
        img_edges = img_edge(img)
        
        #finding the contours in the image
        contours, hierarchy = cv.findContours(img_edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        #for object det
        start_0 = time.time()
        objects = find_objects(contours,n_ob,img.shape[0],img.shape[1])
        end_0 = time.time()
        m = len(objects)
        #showing the number of objects found
        cv.putText(img_original,str(m)+' OBJECTS FOUND',
               (20,50), font, 0.75,(0,0,0),2,cv.LINE_AA)
        for i in range(m):
            gradient = int(255/m)
            #showing the rectangle that contains the object
            p_0 = objects[i][0]
            p_1 = objects[i][1]
            cv.rectangle(img_original,p_0,p_1,(0,gradient*i,255-gradient*i),3) 
        
            #finding the centre of the boject
            x_ob = int((p_0[0]+p_1[0])/2)
            y_ob = int((p_0[1]+p_1[1])/2)
        
            #showing the x and y in the iamge
            cv.putText(img_original,'x='+str(x_ob),
                   (x_ob,y_ob), font, 0.5,(0,gradient*i,255-gradient*i),2,cv.LINE_AA)
            cv.putText(img_original,'y='+str(y_ob),
                   (x_ob,y_ob+20), font, 0.5,(0,gradient*i,255-gradient*i),2,cv.LINE_AA)
        end_1 = time.time()
        t_rend = round(end_1-start_1,4)
        t_det = round(end_0-start_0,4)
        # showing the rendinring time
        cv.putText(img_original,str(t_rend)+'s t_rend',
               (20,80), font, 0.75,(0,0,0),2,cv.LINE_AA)
        # showing the analsyisn and detection time
        cv.putText(img_original,str(t_det)+'s t_det',
               (20,120), font, 0.75,(0,0,0),2,cv.LINE_AA)
        
        #print("The time taken for img_edge and find_obj is:",end-start,"s")
        
   
        #returning the image with the object found
        return img_original
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
         


