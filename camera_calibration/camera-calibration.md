
## Camera Calibration
<br>
**Goal**<br>



- Learn distortions in camera, intrinsic and extrinsic parameters of camera.
- Learn to find these parameters and undistort images.



Due to radial distortion, straight lines appears to be curved. This effect is more as we move away from the center of image.<br> 
For example, one image is shown below, where two edges of a chess board are marked with red lines. But you can see that border is not a straight line and doesn’t match with the red line. All the expected straight lines are bulged out.

![Radial Distortion](calib_radial.jpg "Radial Distortion")


**The distortion is solved as:**<br>
x<sub>{corrected}</sub> = x( 1 + k<sub>1</sub> r<sup>2</sup> + k<sub>2</sub> r<sup>4</sup> + k<sub>3</sub> r<sup>6</sup>) <br> 
y<sub>{corrected}</sub> = y( 1 + k<sub>1</sub> r<sup>2</sup> + k<sub>2</sub> r<sup>4</sup> + k<sub>3</sub> r<sup>6</sup>)
****

Similarly, another distortion is the **tangential distortion** which occurs because image taking lense is not aligned perfectly parallel to the imaging plane.<br> So some areas in image may look nearer than expected. <br>
**Tangential distortion is solved as:**<br>
x<sub>{corrected}</sub> = x + [ 2p<sub>1</sub>xy + p<sub>2</sub>(r<sup>2</sup> + 2x<sup>2</sup>)]
<br> 
y<sub>{corrected}</sub> = y + [p<sub>1</sub>(r<sup>2</sup>+ 2y<sup>2</sup>) +2 p<sub>2</sub>xy]

In short, we need to find five parameters, known as distortion coefficients given by:<br>

Distortion coefficients=(k<sub>1</sub>, k<sub>2</sub>, p<sub>1</sub>, p<sub>2</sub>, k<sub>3</sub>)<br>

In addition to this, we need to find a few more informatios like:<br> 
- Intrinsic parameters
- Extrinsic parameters of a camera.
<br>
Intrinsic parameters are specific to a camera. It includes information like focal length (f<sub>x</sub>,f<sub>y</sub>), optical centers (c<sub>x</sub>, c<sub>y</sub>) etc.<br> 
It is also called camera matrix and depends on the camera only, so once calculated, it can be stored for future purposes. It is expressed as a 3x3 matrix:

camera matrix = $$
\left(\begin{array}{cc} 
f_x & 0 & c_x\\
0 & f_y & c_y\\
0 & 0 & 1
\end{array}\right)
$$ 

Extrinsic parameters corresponds to rotation and translation vectors which translates a coordinates of a 3D point to a coordinate system.

For stereo applications, these distortions need to be corrected first. To find all these parameters, what we have to do is to provide some sample images of a well defined pattern (eg, chess board). <br>We find some specific points in it ( square corners in chess board). <br>We know its coordinates in real world space and we know its coordinates in image. For better results, we need atleast 10 test patterns.


```python
# This folder has sample chessboards images so we will use them

```


```python

```
