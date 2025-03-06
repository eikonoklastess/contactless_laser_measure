# Contacless Laser Measure

## Demo
https://github.com/user-attachments/assets/e5966662-8081-4dab-9422-c60d9edcaefa

## Overview
this tool/program aims to measure object cross-section precisely using a camera and line laser. Currently it is setup to measure most precisely circles radii using arcs but it can be used to estimate anything. This project is uncompleted. Please contact @eikono@proton.me

## Technical Details
To be able to find real 3D coordinates from an image we triangulate the points corresponding to the laser in the image for this we need to know the position of the camera and its intrinsic (focal length, center...) and the equation of the plane corresponding to the laser line. These are obtained in the calibration portion of the software.

![Equation](https://latex.codecogs.com/svg.image?K=%5Cbegin%7Bbmatrix%7Df_x%26s%26c_x%5C%5C0%26f_y%26c_y%5C%5C0%260%261%5Cend%7Bbmatrix%7D)

![image_plan](https://latex.codecogs.com/svg.image?I=%5Cbegin%7Bbmatrix%7DI%280%2C0%29%26I%281%2C0%29%26%5Chdots%26I%28W-1%2C0%29%5C%5CI%280%2C1%29%26I%281%2C1%29%26%5Chdots%26I%28W-1%2C1%29%5C%5C%5Cvdots%26%5Cvdots%26%5Cddots%26%5Cvdots%5C%5CI%280%2CH-1%29%26I%281%2CH-1%29%26%5Chdots%26I%28W-1%2CH-1%29%5Cend%7Bbmatrix%7D)

![Equation](https://latex.codecogs.com/svg.image?%5Cbegin%7Bbmatrix%7Dx%5C%5Cy%5C%5C1%5Cend%7Bbmatrix%7D=%5Cbegin%7Bbmatrix%7D%5Cfrac%7Bu-c_x%7D%7Bf_x%7D%5C%5C%5Cfrac%7Bv-c_y%7D%7Bf_y%7D%5C%5C1%5Cend%7Bbmatrix%7D)
![Equation](https://latex.codecogs.com/svg.image?\begin{bmatrix}x\\y\\1\end{bmatrix}=K^{-1}\begin{bmatrix}u\\v\\1\end{bmatrix})
![Equation](https://latex.codecogs.com/svg.image?P(t)=t\cdot\begin{bmatrix}u\\v\\1\end{bmatrix},\quad t\geq0)

![Equation](https://latex.codecogs.com/svg.image?Ax+By+Cz+D=0)

![Equation](https://latex.codecogs.com/svg.image?A(ut)+B(vt)+C(t)+D=0)

![Equation](https://latex.codecogs.com/svg.image?t=-\frac{D}{Au+Bv+C})


## Features/Usage


## Installing and Running the Project


