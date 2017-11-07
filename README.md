## Lane and Vehicle Detection
**This is not the Repository for the Udacity's Self Driving Car Nanodegree. This is just collection of self sourced Nanodegree projects and resources used are freely available courserwork on Udacity.**<br>
<br>
**Note:** The repository does not contain any training images. You have to download and unzip the image datasets of vehicles and non-vehicles provided by Udacity and place them in appropriate directories on your own.<br>
Please find all the Challanges, Datasets and other Open Sourced Stuffs by Udacity on Self Driving Car [here](https://github.com/udacity/self-driving-car)<br>
### Overview
Detect lanes and Vehicles using computer vision and Deep Learning techniques. This project is based on the format and open sourced codes from [Udacity Self-Driving Car Nanodegree](https://www.udacity.com/drive), and much of the code is leveraged from the provided Jupyter Notebooks.
### Dependencies

    - Numpy
    - cv2
    - Matplotlib
    - Pickle
    

![Self Driving Car-Vehicles and Lane Detection](https://github.com/akhilesh-k/Lane-and-Vehicles-Detection/blob/master/out.gif)

**The following steps were performed for lane detection:**

   - Compute the camera calibration matrix and distortion coefficients with a given set of chessboard images.
   - Apply a distortion correction to raw images.
   - Use color transforms, gradients to create a thresholded binary image.
   - Apply a perspective transform to rectify binary image ("birds-eye view").
   - Detect lane pixels and fit to find the lane boundary.
   - Determine the curvature of the 
