'''
This file takes in the NIST fingerpint database and trains a model to identify and extract minutia from the fingerprints.  

'''
import cv2
import numpy as np
from skimage import io

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
    distance_threshold = 30
    success_threshold = 250

    #count the number of successful matches
    success = [i for i in matches if i.distance < distance_threshold]

    print(f"Successful minutia features matched count: {len(success)}")

    #compare the images based on the previous threshold of ERR
    if len(success) > success_threshold:
        print("The fingerprints are likely the same")
    else:
        print("The fingerprints are likely different")


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
    distance_threshold = 30
    success_threshold = 1000

    #count the number of successful matches
    success = [i for i in matches if i.distance < distance_threshold]

    print(f"Successful minutia features matched count: {len(success)}")

    #compare the images based on the previous threshold of ERR
    if len(success) > success_threshold:
        print("The fingerprints are likely the same")
    else:
        print("The fingerprints are likely different")

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

    print(f"Successful minutia features matched count: {len(success)}")

    #compare the images based on the previous threshold of ERR
    if len(success) > success_threshold:
        print("The fingerprints are likely the same")
    else:
        print("The fingerprints are likely different")

def main():
    ref_path = "NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0001_01.png"
    sub_path = "NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0002_05.png"

    #orb(ref_path, ref_path)
    #sift(ref_path, ref_path)
    fast(ref_path, sub_path)

if __name__ == "__main__":
    main()

