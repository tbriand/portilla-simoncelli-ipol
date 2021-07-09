 The Portilla-Simoncelli Texture Model: Towards Understanding the Early Visual Cortex
-------------------------------------------------------------------------------------------

*******
SUMMARY
*******

This program implements the Portilla & Simoncelli pyramid-based texture analysis/synthesis algorithm.
It performs texture synthesis by iteratively imposing high-order statistical constraints on local image features.

Reference articles:

[1] J. Portilla and E. Simoncelli, A parametric texture model based on joint statistics of complex wavelet coefficients,
    International journal of computer vision, 40 (2000), pp. 49–70.

[2] T. Briand, Reversibility Error of Image Interpolation Methods: Definition and Improvements,
    Image Processing On Line, 9 (2019), pp. 360–380. https://doi.org/10.5201/ipol.2019.277

This program is part of an IPOL publication:
https://doi.org/10.5201/ipol.2021.324


*******
AUTHORS
*******

Thibaud Briand <briand.thibaud@gmail.com>
Independent researcher

Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
Department of Systems and Computational Biology, Albert Einstein College of Medicine


*******
VERSION
*******

Version 1.00, released on 2021-07-08


*******
LICENSE
*******

This program is free software: you can use, modify and/or redistribute it under the terms of the simplified BSD License.
You should have received a copy of this license along this program. If not, see
<http://www.opensource.org/licenses/bsd-license.html>.

Copyright (C) 2021, Thibaud Briand <briand.thibaud@gmail.com>
Copyright (C) 2021, Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
All rights reserved.


***********
COMPILATION
***********

Required environment: Any unix-like system with a standard compilation environment (make and, C and C++ compilers)

Required libraries: libpng, lipjpeg, libtiff, libfftw3, libgomp

Compilation instructions: run "make" to produce an executable "portilla_simoncelli"


*****
USAGE
*****

The program reads an input image, take some parameters and produce a texture image.
The meaning of the parameters is thoroughly discussed on the accompanying IPOL article.
Usage instructions:

  <Usage>: portilla_simoncelli input output [OPTIONS]

  OPTIONS:
  --------

    The optional parameters are:
    -s,          Specify the number of scales (by default 4)
    -k,          Specify the number of orientations (by default 4)
    -i,          Specify the number of iterations (by default 50)
    -N,          Specify the size Na of the neighborhood for the auto-correlation adjustment (by default 7)
    -v,          Activate verbose mode

    Size parameters:
    -x,          Specify the output width (by default the input width)
    -y,          Specify the output height (by default the input height)
    The output size is required to be a multiple of 2^(P + 1). Otherwise a crop is performed
    -c,          Save the cropped input (1) or not (0) (default 0)

    Initial noise parameters:
    -g,          Specify the noise seed (by default 0)
    -n,          Filename of the initial noise

    Adjusment of the constraints parameters:
    -C1,         Adjust marginal statistics (1) or not (0) (by default 1)
    -C2,         Adjust auto-correlation (1) or not (0) (by default 1)
    -C3,         Adjust magnitude correlation (1) or not (0) (by default 1)
    -C4,         Adjust real correlation (1) or not (0) (by default 1)
    -o,          Write the statistics evolution in a file (1) or not (0) (by default 0)

    Interpolation of two input textures:
    -t, 	       Filename of the second input texture
    -w, 	       Specify the weight of the first texture (by default 0.5)

    Edge handling parameters:
    -e,          Use periodic plus smooth decomposition (1) or not (0) (default 0)
    -a,          Add the smooth component (1) or not (0) (default 0)

    Grayscale conversion (average of the channels):
    -b,          Use grayscale mode (1) or not (0) (default 0)

Execution examples:

  1. Default parameters:

   >portilla_simoncelli data/sample.png output.png

  Remark: The generated texture output.png must match data/sample_output.png

  2. Using 3 scales, 5 orientations and the verbose mode:

   >portilla_simoncelli data/sample.png output.png -s 3 -k 5 -v


*************
LIST OF FILES
*************

makefile    : File for the compilation of the program
README.txt  : This file
License.txt : License file

Source files are in the src/ directory:
analysis.[hcpp]             : Functions for analyzing an image and computing its statistics (Line 1 and Line 2 of Algorithm 5)
constraints.[hcpp]          : Functions for computing and adjusting the constraints (Appendix A)
filters.[hcpp]              : Functions for computing the filters of the pyramid (Section 2.1)
pca.[hcpp]                  : Functions for applying the direct or inverse PCA transform (Appendix B.1)
periodic_plus_smooth.[hcpp] : Functions for computing the periodic plus smooth decomposition of an image (see [2])
portilla_simoncelli.cpp     : Main algorithm to read the command line parameters
ps_lib.[cpph]               : Main function for computing a texture from one or two samples (Algorithm 5) and functions for interpolating textures
pyramid.[cpph]              : Functions for computing the pyramid decomposition of an image (Algorithm 2)
synthesis.[hcpp]            : Functions for the iterative synthesis of the texture given the summary statistics (Line 3 to Line 9 of Algorithm 5 and Algorithm 4)
toolbox.[hcpp]              : Several utility functions (Fourier related)
zoom_bilinear.[hcpp]        : Functions for computing the bilinear zoom of an image (see [2])

External programs are in the external/ directory:
iio.[hc]                    : Functions for reading and writing images. Taken from:
                              https://github.com/mnhrdt/iio
mt19937ar.[hc]              : Functions for generating random numbers. Taken from:
                              https://github.com/clibs/mt19937ar
The Eigen library (http://eigen.tuxfamily.org) is provided in the Eigen_library/ directory
This library is used for the matrix computations and for solving polynomial equations.

Test data is provided in the data/ directory:
sample.png                  : Example of input texture
sample_output.png           : Example of generated texture
