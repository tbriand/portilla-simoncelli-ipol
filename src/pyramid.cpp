// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2021, Thibaud Briand <briand.thibaud@gmail.com>
// Copyright (C) 2021, Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
// All rights reserved.

#include <stdlib.h>

#include "ps_lib.h"
#include "toolbox.h"

// Allocate memory for the multi-scale pyramid decomposition
// For the synthesis (option = 0): the only low-band computed is the last one
// For the analysis (option = 1): the low-bands are computed for each scale (1+N_pyr low-bands)
void allocate_pyramid(pyramidStruct *pyramid, int N_pyr, int N_steer,
                      int nx, int ny, int nz, int option)
{
  pyramid->steered  = (fftwf_complex **)
    fftwf_malloc((N_pyr*N_steer)*sizeof(fftwf_complex *));
  pyramid->highband = (float *) malloc( nx*ny*nz* sizeof(float));
  pyramid->lowband = (float **) malloc( (1 + N_pyr) * sizeof(float *));

  int sx = nx;
  int sy = ny;
  for(int i = 0; i < N_pyr; i++) {
    for(int j = 0; j < N_steer; j++)
      pyramid->steered[N_steer*i+j] = (fftwf_complex *)
        fftwf_malloc( sx*sy*nz* sizeof(fftwf_complex));
    if ( option )
      pyramid->lowband[i] = (float *) malloc( sx*sy*nz*sizeof(float *));
    sx /= 2;
    sy /= 2;
  }
  pyramid->lowband[N_pyr] = (float *) malloc( sx*sy*nz* sizeof(float));

  // plans for the FFTs and iFFTs
  pyramid->plan = (fftwf_plan *) fftwf_malloc( (1 + N_pyr)*sizeof(fftwf_plan));
  pyramid->iplan = (fftwf_plan *) fftwf_malloc( (1 + N_pyr)*sizeof(fftwf_plan));
  pyramid->in_plan = (fftwf_complex *)
    fftwf_malloc( nx*ny* sizeof(fftwf_complex));
  pyramid->out_plan = (fftwf_complex *)
    fftwf_malloc( nx*ny* sizeof(fftwf_complex));
}

// Free memory for the multi-scale pyramid decomposition
// For the synthesis (option = 0): the only low-band computed is the last one
// For the analysis (option = 1): the low-bands are computed for each scale (1+N_pyr low-bands)
void free_pyramid(pyramidStruct pyramid, int N_pyr, int N_steer, int option)
{
  int i;

  free(pyramid.highband);
  for(i = 0; i < N_pyr*N_steer; i++)
    fftwf_free(pyramid.steered[i]);
  fftwf_free(pyramid.steered);
  if ( option )
    for (i = 0; i < N_pyr; i++)
      free(pyramid.lowband[i]);
  free(pyramid.lowband[N_pyr]);
  free(pyramid.lowband);

  // plans for the FFTs and iFFTs
  for(i = 0; i < 1+N_pyr; i++) {
    fftwf_destroy_plan(pyramid.plan[i]);
    fftwf_destroy_plan(pyramid.iplan[i]);
  }
  fftwf_free(pyramid.plan);
  fftwf_free(pyramid.iplan);
  fftwf_free(pyramid.in_plan);
  fftwf_free(pyramid.out_plan);
}

// Compute the multi-scale pyramid decomposition of an image (see Section 2.4)
// The filtering are performed in the Fourier domain (see Section 2.3)
// Computations are saved by avoiding the unnecessary DFT and iDFT computations
// between successive steps
// option = 1 <--> analysis of the sample
// option = 0 <--> synthesis of the texture
void create_pyramid(pyramidStruct pyramid, const imageStruct image,
                    const filtersStruct filters, const paramsStruct params,
                    int option)
{
  int i, j, l, sx, sy;

  // parameters
  int N_steer = params.N_steer;
  int N_pyr = params.N_pyr;
  int nx = image.nx;
  int ny = image.ny;
  int nz = image.nz;

  // allocate memory
  fftwf_complex *fft_tmp = (fftwf_complex *)
    fftwf_malloc( nx*ny*nz* sizeof(fftwf_complex));
  fftwf_complex *fft_tmp2 = (fftwf_complex *)
    fftwf_malloc( nx*ny*nz* sizeof(fftwf_complex));

  // Compute the fft of the input
  do_fft_plan_real(pyramid.plan[0], pyramid.out_plan, pyramid.in_plan,
                   fft_tmp, image.image, nx*ny, nz);

  // set the mean to 0
  for (l = 0; l < nz; l++)
    fft_tmp[l*nx*ny][0] = fft_tmp[l*nx*ny][1] = 0.0;

  // Compute the high-frequency residual (Line 1)
  pointwise_complexfloat_multiplication(fft_tmp2, fft_tmp, filters.highpass0,
                                        nx*ny, nz);
  do_ifft_plan_real(pyramid.iplan[0], pyramid.out_plan, pyramid.in_plan,
                    pyramid.highband, fft_tmp2, nx*ny, nz);

  // Compute the low-frequency band (Line 2)
  pointwise_complexfloat_multiplication(fft_tmp, fft_tmp, filters.lowpass0[0],
                                        nx*ny, nz);
  if ( option ) // analysis case
    do_ifft_plan_real(pyramid.iplan[0], pyramid.out_plan, pyramid.in_plan,
                      pyramid.lowband[0], fft_tmp, nx*ny, nz);

  // recursive loop to compute the oriented sub-bands (Line 3 to Line 7)
  sx = nx;
  sy = ny;
  // Loop over the scales (Line 3)
  for(i = 0; i < N_pyr; i++) {
    // loop over the steered filters (Line 4)
    for(j = 0; j < N_steer; j++) {
      // apply the filters (Line 5)
      pointwise_complexfloat_multiplication(fft_tmp2, fft_tmp,
                                            filters.steered[j + i*N_steer],
                                            sx*sy, nz);
      do_ifft_plan(pyramid.iplan[i], pyramid.out_plan, pyramid.in_plan,
                   pyramid.steered[j + i*N_steer], fft_tmp2, sx*sy, nz);
    }

    // moving to next scale
    sx /= 2;
    sy /= 2;

    // down-sample (Line 6)
    // Note that this cannot be in-place for color images
    downsampling(fft_tmp2, fft_tmp, sx, sy, nz);

    // low-pass filtering (Line 7)
    pointwise_complexfloat_multiplication(fft_tmp, fft_tmp2,
                                          filters.lowpass0[i+1], sx*sy, nz);

    // compute the low-band if analysis (option = 1) or last scale (Line 8)
    if ( option || i == N_pyr - 1) {
      // if analysis then compute the ifft of the current low-band
      // if synthesis then compute the ifft of the last low-band
      do_ifft_plan_real(pyramid.iplan[i+1], pyramid.out_plan, pyramid.in_plan,
                        pyramid.lowband[i+1], fft_tmp, sx*sy, nz);
    }
  }

  // Free memory
  fftwf_free(fft_tmp);
  fftwf_free(fft_tmp2);
}
