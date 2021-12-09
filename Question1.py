import os
import numpy as np
import cv2

folder_list = os.listdir("EKG")
folder_list

for folder in folder_list:
    image_list = os.listdir("EKG/"+folder)
    for name in image_list:
        image_name = "EKG/" + folder + "/" + name
        image = cv2.imread(image_name)
        
        crop_list = []
        
        image_crop = image[370:510, 114:422]
        crop_list.append(image_crop)
        
        image_crop = image[370:510, 422:730]
        crop_list.append(image_crop)
        
        image_crop = image[370:510, 730:1038]
        crop_list.append(image_crop)
        
        image_crop = image[370:510, 1038:1346]
        crop_list.append(image_crop)
        
        image_crop = image[510:650, 114:422]
        crop_list.append(image_crop)
        
        image_crop = image[510:650, 422:730]
        crop_list.append(image_crop)
        
        image_crop = image[510:650, 730:1038]
        crop_list.append(image_crop)
        
        image_crop = image[510:650, 1038:1346]
        crop_list.append(image_crop)
        
        image_crop = image[650:790, 114:422]
        crop_list.append(image_crop)
        
        image_crop = image[650:790, 422:730]
        crop_list.append(image_crop)
        
        image_crop = image[650:790, 730:1038]
        crop_list.append(image_crop)
        
        image_crop = image[650:790, 1038:1346]
        crop_list.append(image_crop)
        print(image_name + " " + str(np.array(crop_list).shape))