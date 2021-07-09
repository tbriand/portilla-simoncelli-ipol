// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2021, Thibaud Briand <briand.thibaud@gmail.com>
// Copyright (C) 2021, Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
// All rights reserved.

/* The original code was taken from:
 * T. Briand, Reversibility Error of Image Interpolation Methods: Definition and
 * Improvements, Image Processing On Line, 9 (2019), pp. 360–380.
 * https://doi.org/10.5201/ipol.2019.277
 * See Section 3.2 for more details.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include "toolbox.h"

/* M_PI is a POSIX definition */
#ifndef M_PI
/** macro definition for Pi */
#define M_PI 3.14159265358979323846
#endif                          /* !M_PI */

// Compute the jumps at the boundary of the image (Equation 17 and 18)
static void jumps(float *out, const float *in, int w, int h, int pd)
{
  // initialization
  for (int i = 0; i < w*h*pd; i++)
      out[i] = 0;

  // loop over the channels
  for (int l = 0; l < pd; l++) {
    // horizontal jumps
    for (int j = 0; j < h; j++) {
      out[j*w + l*w*h] = in[j*w + l*w*h] - in[j*w + w-1 + l*w*h];
      out[j*w + w-1 + l*w*h] -= in[j*w + l*w*h] - in[j*w + w-1 + l*w*h];
    }
    // vertical jumps
    for (int i = 0; i < w; i++) {
      out[i + l*w*h] += in[i + l*w*h] - in[(h-1)*w + i + l*w*h];
      out[(h-1)*w + i + l*w*h] -= in[i + l*w*h] - in[(h-1)*w + i + l*w*h];
    }
  }
}

// Compute the smooth component of an image using Fourier computations (Equation 22)
static void compute_smooth_component(float *smooth, const float *in, int w,
                                     int h, int pd)
{
  // allocate memory
  float *v = (float *) malloc(w*h*pd*sizeof*v);
  fftwf_complex *vhat = (fftwf_complex*) malloc(w*h*pd*sizeof*vhat);

  // compute jumps (Equation 17 and 18)
  jumps(v, in, w, h, pd);

  // compute the fft of jumps
  do_fft_real(vhat, v, w, h, pd);

  float tmp;
  float factorh = 2*M_PI/h;
  float factorw = 2*M_PI/w;
  for (int j = 0; j < h; j++)
    for (int i = 0; i < w; i++) {
      tmp = 1.0/(4-2*cos(j*factorh)-2*cos(i*factorw));
      for (int l = 0; l < pd; l++) {
        vhat[j*w+i+l*w*h][0] = vhat[j*w+i+l*w*h][0]*tmp;
        vhat[j*w+i+l*w*h][1] = vhat[j*w+i+l*w*h][1]*tmp;
      }
    }

  // set the mean to 0
  for (int l = 0; l < pd; l++) {
    vhat[l*w*h][0] = 0.0;
    vhat[l*w*h][1] = 0.0;
  }

  // compute the ifft
  do_ifft_real(smooth, vhat, w, h, pd);

  // free memory
  free(v);
  free(vhat);
}

// Compute the periodic plus smooth decomposition of an image (Section 3.2)
void periodic_plus_smooth_decomposition(float *periodic, float *smooth,
                                        const float *in, int w, int h, int pd)
{
  // compute the smooth component (Equation 22)
  compute_smooth_component(smooth, in, w, h, pd);

  // compute the periodic component
  for (int i = 0; i < w*h*pd; i++)
    periodic[i] = in[i] - smooth[i];
}
