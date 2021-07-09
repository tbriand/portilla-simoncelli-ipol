// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2021, Thibaud Briand <briand.thibaud@gmail.com>
// Copyright (C) 2021, Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
// All rights reserved.

/* ps_lib.cpp contains the main function for computing a texture from one or
 * two samples (Algorithm 5) and functions for interpolating textures
 */

#ifndef PS_LIBH
#define PS_LIBH

#include <fftw3.h>

// comment to disable parallel FFT multi-threading
#define FFTW_NTHREADS

// define constants
#define N_PIXELSTATS 6
#define N_PIXELSTATSPCA 2
#define N_SMALLEST 5

// define structures
struct imageStruct {
  float *image;
  int nx;
  int ny;
  int nz;
};

struct paramsStruct {
  int N_steer;
  int N_pyr;
  int N_iteration;
  int Na;
  int noise;
  int edge_handling;
  int add_smooth;
  int cmask[4];
  int verbose;
  float interpWeight;
  int statistics;
};

struct filtersStruct {
  float *highpass0;
  float **lowpass0;
  float **steered;
  float **mask;
  int *size;
};

struct pyramidStruct {
  fftwf_complex **steered;
  float *highband;
  float **lowband;
  fftwf_complex *in_plan;
  fftwf_complex *out_plan;
  fftwf_plan *plan;
  fftwf_plan *iplan;
};

struct statsStruct {
  // for all images
  float *pixelStats;
  float *skewLow;
  float *kurtLow;
  float *varHigh;
  float **autoCorLow;
  float *magMeans;
  float **autoCorMag;
  float **cousinMagCor;
  float **parentMagCor;
  float **parentRealCor;
  // for color images only
  float *pixelStatsPCA;
  float **cousinRealCor;
  float *autoCorPCA;
  float covariancePCA[3][3];
  float eigenVectorsPCA[3][3];
  float eigenValuesPCA[3];
};

void ps(imageStruct texture, imageStruct sample, imageStruct sample2,
        paramsStruct params);

#endif
