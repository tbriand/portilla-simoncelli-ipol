// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2021, Thibaud Briand <briand.thibaud@gmail.com>
// Copyright (C) 2021, Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
// All rights reserved.

/* zoom_bilinear.cpp contains the function for computing the bilinear zoom
 * of an image
 * The original code was taken from:
 * T. Briand, Reversibility Error of Image Interpolation Methods: Definition and
 * Improvements, Image Processing On Line, 9 (2019), pp. 360–380.
 * https://doi.org/10.5201/ipol.2019.277
 */

#ifndef ZOOM_BILINEAR_H
#define ZOOM_BILINEAR_H

void zoom_bilinear(float *X, int W, int H, float *x, int w, int h, int pd);

#endif
