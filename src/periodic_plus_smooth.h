// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2021, Thibaud Briand <briand.thibaud@gmail.com>
// Copyright (C) 2021, Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
// All rights reserved.

/* periodic_component.c contains the functions for computing the periodic
 * plus smooth decomposition of an image.
 * The original code was taken from:
 * T. Briand, Reversibility Error of Image Interpolation Methods: Definition and
 * Improvements, Image Processing On Line, 9 (2019), pp. 360–380.
 * https://doi.org/10.5201/ipol.2019.277
 */

#ifndef PERIODIC_PLUS_SMOOTH_H
#define PERIODIC_PLUS_SMOOTH_H

void periodic_plus_smooth_decomposition(float *periodic, float *smooth,
                                        const float *in, int w, int h, int pd);

#endif
