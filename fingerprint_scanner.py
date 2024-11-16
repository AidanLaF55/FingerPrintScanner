'''
This file takes in the NIST fingerpint database and trains a model to identify and extract minutia from the fingerprints.
It uses the cv2 library and tensorflow to try to train a model
Author Aidan LaFond  

'''
import cv2
import numpy as np
from skimage import io
import os
import glob
import random
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt



#This function uses the Oriented Fast and Rotated Brief algorithm from the OpenCV library
#https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
#https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
def orb(ref_image_path, sub_image_path):

    #take in the images
    ref_image = io.imread(ref_image_path, as_gray=True)
    sub_image = io.imread(sub_image_path, as_gray=True)

    #initialize the orb
    orb = cv2.ORB_create()

    #use the orb library to calculate the keypoints and descriptors of the fingerprints
    keypoints_ref, descriptors_ref = orb.detectAndCompute(ref_image, None)
    keypoints_sub, descriptors_sub = orb.detectAndCompute(sub_image, None)

    #draw the keypoints on the images to show differences/similarity
    #ref_with_keypoints = cv2.drawKeypoints(ref_image, keypoints_ref, None, color=(0,255,0))
    #sub_with_keypoints = cv2.drawKeypoints(sub_image, keypoints_sub, None, color=(0,255,0))

    #display the images with the keypoints drawn
    #cv2.imshow("Reference Image Keypoints of Minutia", ref_with_keypoints)
    #cv2.imshow("Subject Image Keypoints of Minutia", sub_with_keypoints)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #initialize the brute force matching object from cv2
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    #attempt to match the descriptors calculated from both images
    matches = brute_force.match(descriptors_ref, descriptors_sub)

    #sort the matches based on distance, closer matches will be placed first
    matches = sorted(matches, key=lambda x: x.distance)

    #implement a threshold that will represent the equal error rate of the matches
    #this threshold can and should be tested/modified for accuracy
    distance_threshold = 100
    success_threshold = 250

    #count the number of successful matches
    success = [i for i in matches if i.distance < distance_threshold]

    print(f"ORB Algorithm ------> Successful minutia features matched count: {len(success)}")

    return len(matches)


#This function uses the Scale Invariant Feature Transform algorithm from the opencv library
#https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
#https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html 
def sift(ref_image_path, sub_image_path):

    #take in the images
    ref_image = io.imread(ref_image_path, as_gray=True)
    sub_image = io.imread(sub_image_path, as_gray=True)

    #initialize the sift
    sift = cv2.SIFT_create()

    keypoints_ref, descriptors_ref = sift.detectAndCompute(ref_image, None)
    keypoints_sub, descriptors_sub = sift.detectAndCompute(sub_image, None)

    #draw the keypoints on the images to show differences/similarity
    #ref_with_keypoints = cv2.drawKeypoints(ref_image, keypoints_ref, None, color=(0,255,0))
    #sub_with_keypoints = cv2.drawKeypoints(sub_image, keypoints_sub, None, color=(0,255,0))

    #display the images with the keypoints drawn
    #cv2.imshow("Reference Image Keypoints of Minutia", ref_with_keypoints)
    #cv2.imshow("Subject Image Keypoints of Minutia", sub_with_keypoints)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #initialize the brute force matching object from cv2
    #need to use the default L2 matcher for sift rather than hamming distance 
    brute_force = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    #attempt to match the descriptors calculated from both images
    matches = brute_force.match(descriptors_ref, descriptors_sub)

    #sort the matches based on distance, closer matches will be placed first
    matches = sorted(matches, key=lambda x: x.distance)

    #implement a threshold that will represent the equal error rate of the matches
    #this threshold can and should be tested/modified for accuracy
    distance_threshold = 100
    success_threshold = 1000

    #count the number of successful matches
    success = [i for i in matches if i.distance < distance_threshold]

    print(f"SIFT Algorithm ------> Successful minutia features matched count: {len(success)}")

    return len(matches)

#This function makes use of the FAST algorithm and the BRIEF algorithm
#FAST will detect the minutia features
#BRIEF will compute the distances between the detected features.
#https://docs.opencv.org/3.4/dc/d7d/tutorial_py_brief.html
#https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
def fast(ref_image_path, sub_image_path):

    #take in the images
    ref_image = io.imread(ref_image_path, as_gray=True)
    sub_image = io.imread(sub_image_path, as_gray=True)

    #initialize the FAST object and the BRIEF object
    fast = cv2.FastFeatureDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    

    #detect the minutia features with the FAST algorithm
    keypoints_ref = fast.detect(ref_image, None)
    keypoints_sub = fast.detect(sub_image, None)

    #compute the distances with the BRIEF algorithm
    keypoints_ref, descriptors_ref = brief.compute(ref_image, keypoints_ref)
    keypoints_sub, descriptors_sub = brief.compute(sub_image, keypoints_sub)

    
    #draw the keypoints on the images to show differences/similarity
    #ref_with_keypoints = cv2.drawKeypoints(ref_image, keypoints_ref, None, color=(0,255,0))
    #sub_with_keypoints = cv2.drawKeypoints(sub_image, keypoints_sub, None, color=(0,255,0))

    #display the images with the keypoints drawn
    #cv2.imshow("Reference Image Keypoints of Minutia", ref_with_keypoints)
    #cv2.imshow("Subject Image Keypoints of Minutia", sub_with_keypoints)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #initialize the brute force matching object from cv2
    #need to use the hamming distance
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    #attempt to match the descriptors calculated from both images
    matches = brute_force.match(descriptors_ref, descriptors_sub)
    matches = sorted(matches, key=lambda x: x.distance)


    #implement a threshold that will represent the equal error rate of the matches
    #this threshold can and should be tested/modified for accuracy
    distance_threshold = 30
    success_threshold = 100

    #count the number of successful matches
    success = [i for i in matches if i.distance < distance_threshold]

    print(f"FAST Algorithm ------> Successful minutia features matched count: {len(success)}")

    return len(matches)



def calculate_eer(y_true, y_scores):
    # Calculate the false positive rate (FPR), true positive rate (TPR), and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Calculate the false rejection rate (FRR)
    frr = 1 - tpr
    
    # Find the point where FAR equals FRR
    eer_threshold = thresholds[np.nanargmin(np.absolute((frr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((frr - fpr)))]

    #print the min, average, and max FRR values for inclusion in the table
    print(f"Minimum FRR: {np.min(frr)}") 
    print(f"Average FRR: {np.mean(frr)}") 
    print(f"Maximum FRR: {np.max(frr)}") 
    print(f"Minimum FAR: {np.min(fpr)}") 
    print(f"Average FAR: {np.mean(fpr)}") 
    print(f"Maximum FAR: {np.max(fpr)}")

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random guess')
    plt.xlim([-0.5, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    return eer, eer_threshold



def load_images():
    #open the file containing all image paths
    ref_base_path = "NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/reference/"
    sub_base_path = "NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/subject/"
    with open("NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/reference.txt", 'r') as ref:
        ref_image_paths = ref.readlines()
    
    with open("NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/subject.txt", 'r') as sub:
        sub_image_paths = sub.readlines()

    #initiliaze variables to hold similarity scores and the base truth table
    y_true = []
    y_scores = []

    #using the test data, we have already trained the model with 0-1500 to get the match threshold values
    for i in range(1500, 2000):
        #go through all of the images, matching reference to subject
        ref_image_path = ref_base_path + ref_image_paths[i].strip()
        sub_image_path = sub_base_path + sub_image_paths[i].strip()

        #calculate the features extracted across the three algorithms
        orb_matches = orb(ref_image_path, sub_image_path)
        sift_matches = sift(ref_image_path, sub_image_path)
        fast_matches = fast(ref_image_path, sub_image_path)
        y_scores.append(orb_matches)
        y_true.append(1 if orb_matches > 175 else 0)
        y_scores.append(sift_matches)
        y_true.append(1 if sift_matches > 200 else 0)
        y_scores.append(fast_matches)
        y_true.append(1 if fast_matches > 200 else 0)
    
    y_scores = np.array(y_scores)
    y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
    #print(f"yscores: {y_scores}\nytrue: {y_true}")
    eer, eer_threshold = calculate_eer(y_true, y_scores)

    #print the EER of the dataset for the table
    print(f"The EER of this dataset is {100 * eer}%")

        

load_images()

