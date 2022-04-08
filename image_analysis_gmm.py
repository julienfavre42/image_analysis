# -*- coding: utf-8 -*-

import numpy as np
from skimage import data, color
from skimage.color import rgb2gray
import skimage.filters as filters
from skimage import exposure
from scipy.ndimage import gaussian_filter
from skimage.filters import median, gaussian
from skimage.morphology import disk
import matplotlib
#matplotlib.use('Agg')
from matplotlib import *
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import cv2
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy import linalg
from matplotlib.colors import LinearSegmentedColormap

phases_number=2    # Number of phases to detect (or at least number of shades of gray to consider...)
selected_phases=phases_number*[1]
# selected_phases=phases_number*[0]     # You can manually unselect all the "phases", and then select specific ones to be detected and computed
# selected_phases[0]=1
# selected_phases[phases_number-1]=1
phase_merge=[]
#phase_merge=[[7, 8]]  # Put here the phases references that you want to merge together - e.g if you know that #i and #j are the same

gmm_type="full" # Gmm covariance type : full’, ‘tied’, ‘diag’, ‘spherical’

def get_threshold(image_):                  # This function proceeds to the GMM setting and detects automatically the thresholds
    gray_scale=np.linspace(0, 256, 257)
    image_values=image_.flatten()
    gmm = GaussianMixture(n_components = phases_number, covariance_type=gmm_type, max_iter=100000, n_init=5, tol=1e-3)
    gmm.fit(np.reshape(image_values, (image_values.shape[0], 1)))
    gauss_mixt = np.array([p * norm.pdf(gray_scale, mu, sd) for mu, sd, p in zip(gmm.means_.flatten(), np.sqrt(gmm.covariances_.flatten()), gmm.weights_)])
    gauss_mixt_t = np.sum(gauss_mixt, axis = 0)
    
    fig= plt.subplots(figsize=(10, 10))
    plt.hist(image_.ravel(), bins=256, range=(0.0, 256.0), density=True, histtype='step', label="gray_level")
    plt.plot(gray_scale, gauss_mixt_t, label='total')
    for k in range(0, gauss_mixt.shape[0]):
        plt.plot(gray_scale, gauss_mixt[k], label=str(k+1))
    plt.legend()
    plt.savefig(fname+'_gmm_gray_levels.png')
    plt.close()
    
    unsorted_phase_levels=np.zeros([gray_scale.shape[0]])
    for k in range(0, gray_scale.shape[0]):
       unsorted_phase_levels[k]=list(gauss_mixt[:,k]).index(np.max(gauss_mixt[:,k]))+1      # Identify phase index for gray levels
    levels=np.unique(unsorted_phase_levels)
    unsorted_levels=list(dict.fromkeys(unsorted_phase_levels))
    phase_levels=0*unsorted_phase_levels
    for k in range(0, gray_scale.shape[0]):
        phase_levels[k]=levels[unsorted_levels.index(unsorted_phase_levels[k])]     # Sorting phases index
    for k in range(0, len(phase_merge)):
        phase_levels[phase_levels==phase_merge[k][0]]=phase_merge[k][1]
    for k in range(0, gray_scale.shape[0]-1):
        image_[((image_>=gray_scale[k])*(image_<gray_scale[k+1]))]=phase_levels[k]      # Make image masks
    image_[(image_==gray_scale[-1])]=phase_levels[-1]
    return (image_)

def plot_single_channel(image_, color_, k):
    fig= plt.subplots(figsize=(10, 10))
    cmap_ = LinearSegmentedColormap.from_list("phase_cmap", [(0,0,0), color_], N=2)
    plt.imshow(image_, cmap=cmap_)
    plt.savefig(fname+'phase_'+str(k)+'.png')
    plt.close()

def normalize_image(image, low_lim, high_lim):      # This function "renormalizes" the gray level to its maximal amplitude before 0 and 255
    image_min=np.amin(image)
    image_max=np.amax(image)
    image=low_lim+(((image-image_min)*(high_lim-low_lim))/(image_max-image_min))
    return image

def increase_contrast(image_):      # You are supposed to adjust the options here for the treatment of the picture, to help the segmentation
    # Normalization
    # image_=normalize_image(image_, 0, 255)  
    # # Correct some possible variations of brightness  # That is especially useful for OM pictures, having variations of brightness that makes segmentation impossible...
    # image_=image_-gaussian_filter(image_, 100)
    # image_=normalize_image(image_, 0, 255)
    # image_=image_-gaussian_filter(image_, 50)
    # image_=normalize_image(image_, 0, 255)
    # Filter
    image_=gaussian_filter(image_, 1)
    image_ = median(image_, disk(3)) # Median filter
    # Normalization
    # image_=normalize_image(image_, 0, 255)  
    return image_

# Import the data
filename = "image.png"              # Put the name of your image here
fname=filename.replace(".png","")
image = cv2.imread(filename)

gray_img_initial=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray_img=increase_contrast(gray_img_initial)
img = cv2.merge((gray_img_initial,gray_img_initial,gray_img_initial)).astype("int")
phases_mask=get_threshold(gray_img.copy())

# Calculate phase fraction
phase_fraction=[]
filewrite=open(fname+"phase_fraction.txt",'w')
for k in range(0, phases_number):
    if selected_phases[k]==1:
        fraction=100*np.sum(phases_mask==k+1)/(phases_mask.shape[0]*phases_mask.shape[1])
        phase_fraction.append(fraction)
        fractionstring="Fraction of phase #"+str(k+1)+" = "+str(np.round(fraction,2))+" %\n"
        print(fractionstring)
        filewrite.write(fractionstring)
filewrite.close()

# Detect particles
fig= plt.subplots(figsize=(16, 16))
colorlist=[(255, 0, 0), (125, 125, 0), (0, 125, 125), (125, 0, 125), (200, 50, 0), (0, 0, 255), (50, 200, 0), (0, 0, 255),  (50, 0, 200), (200, 0, 50)] # This is a list of RGB colors, edit it to change the phase color highlighting
phaseplotlist=[]
phasecolorlist=[]
for k in range(0, phases_number):
    phasemask=np.array((phases_mask==k+1), dtype="uint8")
    plot_single_channel(255*phasemask, colorlist[k], k+1)
    if selected_phases[k]==1:
        contours, heirarchy = cv2.findContours(255*phasemask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, colorlist[k], 2)            # Contours are added to highlight the detected phases
        phaseplotlist.append(255*phasemask)
        phasecolorlist.append(colorlist[k])
plt.imshow(img)
for k in range(0, len(phaseplotlist)):
    phase_cmap = LinearSegmentedColormap.from_list("phase_cmap", [phasecolorlist[k], (0,0,0)], N=2)
    masked = np.ma.masked_where(phaseplotlist[k]==0, phaseplotlist[k])
    plt.imshow(masked, alpha=0.1, cmap=phase_cmap, interpolation='none')       # Plots some colors on the detected phases
plt.savefig(fname+'_phase_detection.png')
plt.close()
