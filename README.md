# ImageCoregistration

A simple Image regestring method based on SIFT descriptors and RANSAC algorithm for filtering out the outliers.

# Requirements
* Python3
* numpy
* opencv
* sklearn

# Setup
```
git clone https://github.com/ily-R/ImageCoregistration.git
cd ImageCoregistration
```

# Usage
Very easy to use. Put the images you want to register, the target and the source(say "img1.jpg" and "img2.jpg") in `data\` folder and run the follwing.

```
python image_register.py img1.jpg img2.jpg --sift
```

# Arguments
This is the full set of system arguments supported.
```
python image_register.py img1.jpg img2.jpg -s 0.5 --sift -r

```
* `img1.jpg img2.jpg` are positional arguments representing the names of the target and source image.
* `--scale 0.5` or abbreviation `-s 0.5` is the scale by which we resize the input images. **default** = 0.1.
* `--sift` a boolean flag. If *True* use Harris detectors then sift descriptor. If *False* the images will be displayed and the user has to click on both images to selected the prefered landmarks to be passed to SIFT.**default** = *False*.
* `--rasnac` or abbreviation `-r` a boolean flag. If *True* use RANSAC algorithm to select only the inliers for the affine matrix estimation step. Useful when the images have a lot of similar pattern in different positions in the image. **default** = *False*.

# Notes
* `result\` folder contains the source and target images alligned. Additionally, if *True* flags are passed in inner functions, the folder will contain other results.
  * if `save = True` in ```display_matches(target, source, lmk1, lmk2, name="matches, save=False")``` it will save both images horizontally stacked with landmarks matched.
  * if `save = True` in ```display_matches(target, source, outliers1, outliers2, name="matches_removed_by_RANSAC", num=5, save=True)``` it will save and image that showes the landmarks rejected by RASNAC.
  * if `save = True` in ```mi = mutual_inf(warped, target_w, verbose=True)``` it will save the joint-histogram of warped-source image and the target image.

# Testing
### Original images:
We can see that there a slight translation upward with probably small rotation to the left.

<p align="center">
  <img src="https://github.com/ily-R/ImageCoregistration/blob/master/readmeImages/original.gif?raw=true" alt="capture reconstruction"/>
</p>

### Floating the source image:
Applying the registation on the source image, we see how it floats from its original position to another upward. This is done to assure that most pixels are alligned to the target ones.

<p align="center">
  <img src="https://github.com/ily-R/ImageCoregistration/blob/master/readmeImages/source_moving.gif?raw=true" alt="capture reconstruction"/>
</p>

### Result:
Floating the source image we see how perfectly are the images aligned. To not get distracted by the upper border moving, put the tip of the mouse on any position and verify that the mouse tip stands steel. If you do this in the above gif, the tip will change positions.

<p align="center">
  <img src="https://github.com/ily-R/ImageCoregistration/blob/master/readmeImages/solution.gif?raw=true" alt="capture reconstruction"/>
</p>

### Features:
The tested image showed a perfect match of features as see below. (showing only 20 out of around 200)
<p align="center">
  <img src="https://github.com/ily-R/ImageCoregistration/blob/master/readmeImages/matches.jpg?raw=true" alt="capture reconstruction"/>
</p>

This is due to the use of `sift_error =0.7` which filter out outliers. Read the doc of ` match(lmk1, lmk2, desc1, desc2, sift_error=0.7)` for more details.
Using RANSAC on top of this result will not change the landmarks selection. To force outliers, we let `sift_error =0.9`. Now after using RANSAC, here's the outliers filtered.

<p align="center">
  <img src="https://github.com/ily-R/ImageCoregistration/blob/master/readmeImages/matches_removed_by_RANSAC.jpg?raw=true" alt="capture reconstruction"/>
</p>
We see that the landmarks are similar locally, but globally they refer to different 3D points.

# Metrics
Two metrics are used plus the reconstruction error:
* Cross-correlation: for the above registration `cc = 0.86086107 ` almost `1`, which is the case when the two images are perfectly alligned. This indicates good registation.
* Multi information: it gives a probabilistic measure on how uncertain we are about the target image in the absence/presence of the warped source image. Below is the joint_histogram of both images. We get this figure because we are working in mono-modality registration. Means that the images contain almost the same range of colors with some noise.

<p align="center">
  <img src="https://github.com/ily-R/ImageCoregistration/blob/master/readmeImages/joint_histogram.jpg?raw=true" alt="capture reconstruction"/>
</p>

# More Testing
Here are some screen shot taken from Google maps satellite above the Eiffel tower

<img width = 400 align="left" src="https://github.com/ily-R/ImageCoregistration/blob/master/data/s1.jpg?raw=true" alt="capture reconstruction">	
<p align="right">
  <img width = 400 src="https://github.com/ily-R/ImageCoregistration/blob/master/data/s2.jpg?raw=true" alt="capture reconstruction"/>
</p>

We confirm also what we said about the joint-histogram in this case, as seen below.

<p align="center">
  <img src="https://github.com/ily-R/ImageCoregistration/blob/master/readmeImages/joint_histogram_paris.jpg?raw=true" alt="capture reconstruction"/>
</p>

Applying the registration to float the source to the target image we get the following:

<p align="center">
  <img src="https://github.com/ily-R/ImageCoregistration/blob/master/readmeImages/paris.gif?raw=true" alt="capture reconstruction"/>
</p>

You can see how the Eiffel tower is moving, which indicates the two images present in the gif come from different perspectives. Which in our case the source and target images.

# Future work:
* Add more robust feature for noisy SAR images like 
[SAR-SIFT](https://hal.archives-ouvertes.fr/hal 00831763/file/preprint_sar_sift_dellinger_hal.pdf)
* Add the spatial information to intensity information in our descriptors to limit the outliers in difficult tasks. Like what Johan et al. did [Here](https://arxiv.org/abs/1807.11599)


                                                                                                                                   
