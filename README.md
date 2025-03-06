# Contacless Laser Measure

## Demo
https://github.com/user-attachments/assets/e5966662-8081-4dab-9422-c60d9edcaefa

## Overview
this tool/program aims to measure object cross-section precisely using a camera and line laser. Currently it is setup to measure most precisely circles radii using arcs but it can be used to estimate anything. This project is uncompleted. Please contact @eikono@proton.me

## Technical Details
To be able to find real 3D coordinates from an image we triangulate the points corresponding to the laser in the image for this we need to know the position of the camera and its intrinsic

![intrinsic](https://latex.codecogs.com/svg.image?K=%5Cbegin%7Bbmatrix%7Df_x%26s%26c_x%5C%5C0%26f_y%26c_y%5C%5C0%260%261%5Cend%7Bbmatrix%7D) 

and the equation of the plane corresponding to the laser line 

![plane](https://latex.codecogs.com/svg.image?Ax+By+Cz+D=0). 

These are obtained in the calibration portion of the software. with the Camera intrinsic we can undistort the camera image to obtain a perfect pinhole camera image 

![image_plan](https://latex.codecogs.com/svg.image?I=%5Cbegin%7Bbmatrix%7DI%280%2C0%29%26I%281%2C0%29%26%5Chdots%26I%28W-1%2C0%29%5C%5CI%280%2C1%29%26I%281%2C1%29%26%5Chdots%26I%28W-1%2C1%29%5C%5C%5Cvdots%26%5Cvdots%26%5Cddots%26%5Cvdots%5C%5CI%280%2CH-1%29%26I%281%2CH-1%29%26%5Chdots%26I%28W-1%2CH-1%29%5Cend%7Bbmatrix%7D)

For each point I(x,y) you can convert them into normalized camera coordinate with that equation:
![equation](https://latex.codecogs.com/png.latex?%5Cbegin%7Bbmatrix%7D%20u%20%5C%5C%20v%20%5C%5C%201%20%5Cend%7Bbmatrix%7D%20=%20%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7Bx%20-%20c_x%7D%7Bf_x%7D%20%5C%5C%20%5Cfrac%7By%20-%20c_y%7D%7Bf_y%7D%20%5C%5C%201%20%5Cend%7Bbmatrix%7D)

![equation](https://latex.codecogs.com/png.latex?%5Cbegin%7Bbmatrix%7D%20u%20%5C%5C%20v%20%5C%5C%201%20%5Cend%7Bbmatrix%7D%20=%20K^{-1}%20%5Cbegin%7Bbmatrix%7D%20x%20%5C%5C%20y%20%5C%5C%201%20%5Cend%7Bbmatrix%7D)

The resulting vector (u,v,1) is the direction vector of the ray in camera coordinate going from the pixel throught the camera centre to the real object. Here's the equation of the ray:

![ray equation](https://latex.codecogs.com/png.latex?%5Cmathbf%7BP%7D(t)%20=%20t%20%5Ccdot%20%5Cbegin%7Bbmatrix%7D%20u%20%5C%5C%20v%20%5C%5C%201%20%5Cend%7Bbmatrix%7D,%20%5Cquad%20t%20%5Cgeq%200)

![Equation](https://latex.codecogs.com/svg.image?A(ut)+B(vt)+C(t)+D=0)

![Equation](https://latex.codecogs.com/svg.image?t=-\frac{D}{Au+Bv+C})


## Features/Usage


## Installing and Running the Project


