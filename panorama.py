import numpy as np
import numpy.linalg as LA
from scipy import signal
from scipy import misc
from scipy import ndimage
import scipy.stats as st
from scipy.ndimage import filters
from scipy.stats import multivariate_normal
from numpy.linalg import norm
import numpy.linalg
import cv2
import time
import os
import math
import pandas as pd
import multiprocessing as MP
import matplotlib.pyplot as plt
import random


# INPUTS: imagename (filename of image)
# OUTPUT: keypoints , descriptors

array_of_img = []

class SIFTCLASS():

    def __init__(self):
        pass


    def detectAndDescribe(self,image, method=None):

    	# detect and extract features from the image
    	if method == 'sift':
    		descriptor = cv2.xfeatures2d.SIFT_create()
    	elif method == 'brisk':
    		descriptor = cv2.BRISK_create()
    	elif method == 'akaze':
    		descriptor = cv2.AKAZE_create()
    	# get keypoints and descriptors
    	(kps, features) = descriptor.detectAndCompute(image, None)

    	return kps, features

    '''
    def match(self,kpA, kpB, dsA, dsB):
        
        ratio = 0.45
        matcher = cv2.BFMatcher()
        rawMatches = matcher.knnMatch(dsA, dsB, 2)
        matches1 = []
        matches2 = []


        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches1.append(m[0])
                matches2.append((m[0].queryIdx , m[0].trainIdx))
        mp = np.zeros((len(matches2),2,2))
        
        for i,idx in enumerate(matches2) :
            mp[i][0] = np.array([int(kpA[idx[0]][1]),int(kpA[idx[0]][0])],dtype='int')
            mp[i][1] = np.array([int(kpB[idx[1]][1]),int(kpB[idx[1]][0])],dtype='int')
        
        return matches1, mp
    '''

    
    def matcher(self, kp1, des1, img1, kp2, des2, img2, threshold):
	    # BFMatcher with default params
	    bf = cv2.BFMatcher()
	    matches = bf.knnMatch(des1,des2, k=2)

	    # Apply ratio test
	    good = []
	    for m,n in matches:
	        if m.distance < threshold*n.distance:
	            good.append([m])

	    matches = []
	    for pair in good:
	        matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

	    matches = np.array(matches)
	    return matches


    def ransac(self, matches, threshold, iters):

    	def homography(pairs):
		    rows = []
		    for i in range(pairs.shape[0]):
		        p1 = np.append(pairs[i][0:2], 1)
		        p2 = np.append(pairs[i][2:4], 1)
		        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
		        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
		        rows.append(row1)
		        rows.append(row2)
		    rows = np.array(rows)
		    U, s, V = np.linalg.svd(rows)
		    H = V[-1].reshape(3, 3)
		    H = H/H[2, 2] # standardize to let w*H[2,2] = 1
		    return H

    	def random_point(matches, k=4):
		    idx = random.sample(range(len(matches)), k)
		    point = [matches[i] for i in idx ]
		    return np.array(point)

    	def get_error(points, H):
		    num_points = len(points)
		    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
		    all_p2 = points[:, 2:4]
		    estimate_p2 = np.zeros((num_points, 2))
		    for i in range(num_points):
		        temp = np.dot(H, all_p1[i])
		        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
		    # Compute error
		    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

		    return errors


    	num_best_inliers = 0

    	for i in range(iters):
	        points = random_point(matches)
	        H = homography(points)
	        
	        #  avoid dividing by zero 
	        if np.linalg.matrix_rank(H) < 3:
	            continue
	            
	        errors = get_error(matches, H)
	        idx = np.where(errors < threshold)[0]
	        inliers = matches[idx]

	        num_inliers = len(inliers)
	        if num_inliers > num_best_inliers:
	            best_inliers = inliers.copy()
	            num_best_inliers = num_inliers
	            best_H = H.copy()
            
    	return best_H, best_inliers


    def drawMatches(self, img1, kp1, img2, kp2, matches):
    	# Create a new output image that concatenates the two images together
    	rows1 = img1.shape[0]
    	cols1 = img1.shape[1]
    	rows2 = img2.shape[0]
    	cols2 = img2.shape[1]

    	out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    	# Place the first image to the left
    	out[:rows1,:cols1,:] = np.dstack([img1])

    	# Place the next image to the right of it
    	out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2])

    	# For each pair of points we have between both images
    	# draw circles, then connect a line between them
    	for x1,y1,x2,y2 in matches:

	        # Draw a small circle at both co-ordinates
	        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
	        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

	        color = np.random.randint(0, high = 256, size = (3,)).tolist()
	        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), color, 1)

    	return out


    def warp(self, left, right, H):

    	def autocrop(image, threshold=0):
    		if len(image.shape) == 3: 
    			flatImage = np.max(image, 2) 
    		else:
    			flatImage = image 
    		assert len(flatImage.shape) == 2 

    		rows = np.where(np.max(flatImage, 0) > threshold)[0] 
    		if rows.size:
    			cols = np.where(np.max(flatImage, 1) > threshold)[0] 
    			image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    		else: 
    			image = image[:1, :1] 

    		return image 

    	h = left.shape[0]
    	w = left.shape[1]*2
    	indy, indx = np.indices((h, w), dtype=np.float32)
    	lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])

    	# warp the coordinates of src to those of true_dst
    	map_ind = H.dot(lin_homg_ind)
    	map_x, map_y = map_ind[:-1]/map_ind[-1]  # ensure homogeneity
    	map_x = map_x.reshape(h, w).astype(np.float32)
    	map_y = map_y.reshape(h, w).astype(np.float32)

    	dst = cv2.remap(right, map_x, map_y, cv2.INTER_LINEAR)
    	#blended = cv2.addWeighted(left, 0.5, dst, 1, 0)
    	dst[0:left.shape[0], 0:left.shape[1]] = left
    	#cv2.imshow('blended.png', dst)
    	#cv2.waitKey()

    	out = autocrop(dst, 175)

    	return out


    def createPanorama(self):
    
        pool = MP.Pool(MP.cpu_count())
        
        output = None
        
        print ('Total %d images' %len(array_of_img))
        
        for i in range(len(array_of_img)):
            
            if i==0:
                continue

            left = array_of_img[i-1]
            right = array_of_img[i]

            if i==1:
                output = left
            
            kplsA , desA = self.detectAndDescribe(output, 'sift')
            kplsB , desB = self.detectAndDescribe(right, 'sift')

            '''
            # Show the interest points
            fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)
            ax1.imshow(cv2.drawKeypoints(left,kpls1,None,color=(0,255,0)))
            ax1.set_xlabel("(a)", fontsize=14)
            ax2.imshow(cv2.drawKeypoints(right,kpls2,None,color=(0,255,0)))
            ax2.set_xlabel("(b)", fontsize=14)
            plt.show()
            '''

            #kpls1 = np.float32([kp.pt for kp in kplsA])
            #kpls2 = np.float32([kp.pt for kp in kplsB])

            matches = self.matcher(kplsA, desA, output, kplsB, desB, right, 0.45)

            match_img = self.drawMatches(output, kplsA, right, kplsB, matches)

            H, inliers = self.ransac(matches, 0.6, 2000)
            print("H",H)

            '''
            match_img = cv2.drawMatches(left, kplsA, right, kplsB, np.random.choice(matches1, 100),
		                           None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            if len(matches1) > 4:
            	ptsA = np.float32([kpls1[m.queryIdx] for m in matches1])
            	ptsB = np.float32([kpls2[m.trainIdx] for m in matches1])
        
            	(cv_H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 4)
            	print(cv_H)
            '''

            if i==1:
                output = left
        
            output = self.warp(output, right, H)
            

        pool.close()
        pool.join()

        return output, match_img


def read_directory(directory_name):
    '''
    filenumber = len([name for name in os.listdir(directory_name) if os.path.isfile(os.path.join(directory_name, name))])
    '''
    for file in sorted(os.listdir(directory_name)):
        if file != '.DS_Store':
            img_name = os.path.join(directory_name, file)
            img = cv2.imread(img_name)
            array_of_img.append(img)

                         
def SIFT(inputname):

    read_directory(inputname)
    
    #use: len(array_of_img) for looping the image, array_of_img[0],
    #array_of_img[1],array_of_img[2],...for processing each image
    #Start SIFT here
    t = time.time()
    sift = SIFTCLASS()
    
    
    array_of_img[0], match_img = sift.createPanorama()
    t = time.time()-t
    print ('cost %d mins %d secs'%(int(t)/60,int(t)%60))
    #End of SIFT here and use imageoutput for your output

    imageoutput = array_of_img[0]
    array_of_img.clear()

    return imageoutput, match_img


def main():

    global array_of_img
    
    path = os.getcwd()+'/'
    
    print (path)
    
    for dirname in os.listdir(path):
    
        dir_path = path + dirname
        
        if os.path.isdir( dir_path ):
            imageout, match_img=SIFT(dirname)
            plt.figure()
            plt.imshow(imageout)
            cv2.imwrite(dirname+'.jpg', imageout)
            cv2.imwrite(dirname+'_match.jpg', match_img)

    
if __name__ == '__main__' :

    main()




