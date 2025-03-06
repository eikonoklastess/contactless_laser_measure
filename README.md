# Contacless Laser Measure

## Demo
https://github.com/user-attachments/assets/e5966662-8081-4dab-9422-c60d9edcaefa

## Overview
this tool/program aims to measure object cross-section precisely using a camera and line laser. Currently it is setup to measure most precisely circles radii using arcs but it can be used to estimate anything. This project is uncompleted. Please contact @eikono@proton.me

## Technical Details
To be able to find real 3D coordinates from an image we triangulate the points corresponding to the laser in the image for this we need to know the position of the camera and its intrinsic (focal length, center...) and the equation of the plane corresponding to the laser line. These are obtained in the calibration portion of the software.

![Equation]([(https://latex.codecogs.com/svg.image?I=\begin{bmatrix}I(0,0)&I(1,0)&\hdots&I(W-1,0)\\I(0,1)&I(1,1)&\hdots&I(W-1,1)\\\vdots&\vdots&\ddots&\vdots\\I(0,H-1)&I(1,H-1)&\hdots&I(W-1,H-1)\\\end{bmatrix})])

## Features/Usage


## Installing and Running the Project


