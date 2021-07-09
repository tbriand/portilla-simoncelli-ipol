// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2021, Thibaud Briand <briand.thibaud@gmail.com>
// Copyright (C) 2021, Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
// All rights reserved.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "ps_lib.h"

/* M_PI is a POSIX definition */
#ifndef M_PI
/** macro definition for Pi */
#define M_PI 3.14159265358979323846
#endif                          /* !M_PI */

// Compute the low-pass filter (Equation 7 or Equation 11)
// factor = 1 <--> L
// factor = 2 <--> L0
static void compute_lowpass(float *out, int nx, int ny, float factor)
{
  float x, x2, y, r;
  int j, k;

  float factorx = 4.0/(nx);
  float factory = 4.0/(ny);
  float ifactor = 1.0/(factor*factor);
  float factorcos = 0.25*M_PI/log(2);

  for(j = 0; j < nx; j++) {
    x = (2*j < nx) ? j*factorx : (j-nx)*factorx;
    x2 = x*x;
    for(k=0; k<ny; k++) {
      y = (2*k < ny) ? k*factory : (k-ny)*factory;
      r = (x2+y*y)*ifactor;
      out[j + k*nx] = (r <= 0.25) + (r > 0.25)*(r < 1)*cos(factorcos*log(4*r));
    }
  }
  out[0] = 1.0;
}

// Compute the high-pass filter (Equation 8 or Equation 12)
// factor = 1 <--> H
// factor = 2 <--> H0
static void compute_highpass(float *out, int nx, int ny, float factor)
{
  float x, x2, y, r;
  int j, k;

  float factorx = 4.0/(nx);
  float factory = 4.0/(ny);
  float ifactor = 1.0/(factor*factor);
  float factorcos = 0.25*M_PI/log(2);

  for(j = 0; j < nx; j++) {
    x = (2*j < nx) ? j*factorx : (j-nx)*factorx;
    x2 = x*x;
    for(k = 0; k < ny; k++) {
      y = (2*k < ny) ? k*factory : (k-ny)*factory;
      r = (x2+y*y)*ifactor;
      out[j + k*nx] = (r >= 1) + (r > 0.25)*(r < 1)*cos(factorcos*log(r));
    }
  }
  out[0] = 0.0;
}

// Compute the steered mask (Equation 19)
static void compute_mask(float *out, int nx, int ny, int steer, int N_steer)
{
  float x, y, theta, theta0;
  int j, k;

  float factorx = 2*M_PI/(nx);
  float factory = 2*M_PI/(ny);

  for(j = 0; j < nx; j++) {
    x = (j < nx/2) ? j*factorx : (j-nx)*factorx;
    for(k = 0; k < ny; k++) {
      if(j == nx/2 || k == ny/2)
        out[j+k*nx] = 1;
      else {
        y = (k < ny/2) ? k*factory : (k-ny)*factory;
        theta = atan2(y,x);
        theta0 = 2*fabs(fmod(theta + 3*M_PI - M_PI*steer/N_steer,2*M_PI) - M_PI);
        out[j + k*nx] = 2*(theta0 < M_PI) + (theta0 == M_PI);
      }
    }
  }
  out[0] = 1.0;
}

// Compute the steered filters (Equation 9)
static void compute_steered(float *out, int nx, int ny, int steer, int N_steer,
                            float alpha)
{
  float x, y, theta, theta2, factor, cosinus_theta;
  float factorx = 2*M_PI/(nx);
  float factory = 2*M_PI/(ny);
  int j, k, p;

  for(j = 0; j < nx; j++) {
    x = (2*j < nx) ? j*factorx : (j-nx)*factorx;
    for(k = 0; k < ny; k++) {
      y = (2*k < ny) ? k*factory : (k-ny)*factory;
      theta = atan2(y,x);
      theta2 = 2*fabs(fmod(theta + 3*M_PI - M_PI*steer/N_steer,2*M_PI) - M_PI);
      factor = cos(theta - M_PI*steer/N_steer);
      cosinus_theta = 1.0;
      for(p = 1; p < N_steer; p++)
        cosinus_theta *= factor;
      out[j+k*nx] = 2*alpha*cosinus_theta*(theta2 < M_PI);
    }
  }
  out[0] = 0.0;
}

// Compute a list containing the sizes of the filters
static void size_filters(int *size, int nx, int ny, int N_pyr)
{
  // initialize
  size[0] = nx;
  size[1] = ny;

  // loop over the scales
  for (int i = 1; i < N_pyr+1; i++) {
    // divide previous size by 2
    size[2*i] = size[2*(i-1)]*0.5;
    size[2*i+1] = size[2*(i-1)+1]*0.5;
  }
}

// Compute the reversibility constant alpha (Equation 10)
static float compute_reversibility(int N_steer)
{
  // initialize sums
  float log_fact_alpha1 = 0.0;
  float log_fact_alpha2 = 0.0;

  // compute the sums
  for(int k = 2; k < N_steer; k++)
    log_fact_alpha1 += log(k);
  for(int k = 2; k < 2*N_steer-1; k++)
    log_fact_alpha2 += log(k);

  // compute log(alpha)
  float log_alpha = (N_steer-1)*log(2) + log_fact_alpha1 - 0.5*(log(N_steer) + log_fact_alpha2);

  return(exp(log_alpha));
}

// Compute the filters used during the multi-scale pyramid decomposition
// See Section 2.2 and Section 2.4 for more details
// option = 1 <--> analysis
// option = 0 <--> synthesis
void compute_filters(filtersStruct *filters, int nx, int ny, int N_pyr,
                     int N_steer, int option)
{
  int ind, i, j, l, sx = 0, sy = 0;

  // compute the sizes of the filters
  filters->size = (int *) malloc( 2 * (1 + N_pyr) * sizeof(int));
  size_filters(filters->size, nx, ny, N_pyr);

  // memory allocation
  filters->lowpass0 = (float **) malloc( (1 + N_pyr) *sizeof(float *));
  filters->highpass0 = (float *) malloc( nx*ny*sizeof(float));
  filters->steered = (float **) malloc( N_pyr*N_steer* sizeof(float *));
  if ( !option ) // synthesis case
    filters->mask = (float **) malloc( N_pyr*N_steer *sizeof(float *));
  for(i = 0; i < N_pyr; i++) {
    sx = filters->size[2*i];
    sy = filters->size[2*i+1];
    filters->lowpass0[i] = (float *) malloc(sx*sy*sizeof(float *));
    for (j = 0; j<N_steer; j++) {
      filters->steered[i*N_steer + j] = (float *) malloc(sx*sy*sizeof(float *));
      if ( !option ) // synthesis case
        filters->mask[i*N_steer + j] = (float *) malloc(sx*sy*sizeof(float *));
    }
  }
  filters->lowpass0[N_pyr] = (float *) malloc(0.25*sx*sy*sizeof(float *));
  float *highpass = (float *) malloc(nx*ny*sizeof(float));

  // compute the high-pass filter H0
  compute_highpass(filters->highpass0, nx, ny, 2);

  // compute the reversibility constant
  float alpha = compute_reversibility(N_steer);

  // compute the other filters of the pyramid
  for(i = 0; i < N_pyr; i++) {
    // update sizes
    sx = filters->size[2*i];
    sy = filters->size[2*i+1];

    // compute the low-pass filter L0
    compute_lowpass(filters->lowpass0[i], sx, sy, 2);

    // compute the high-pass filter for this scale
    compute_highpass(highpass, sx, sy, 1);

    // compute the steered filters for the analysis
    for(j=0; j<N_steer; j++) {
      ind = i*N_steer+j;
      compute_steered(filters->steered[ind],sx, sy, j, N_steer, alpha);
      for(l=0; l<sx*sy; l++)
        filters->steered[ind][l] *= highpass[l];
      if( !option )
        compute_mask(filters->mask[ind], sx, sy, j, N_steer);
    }
  }

  // compute the last low-pass filter L0
  compute_lowpass(filters->lowpass0[N_pyr], sx*0.5, sy*0.5, 2);

  // free memory
  free(highpass);
}

// Free memory for the filters
// option = 1 <--> analysis
// option = 0 <--> synthesis
void free_filters(filtersStruct filters, int N_pyr, int N_steer, int option)
{
  int i;
  free(filters.highpass0);
  for(i = 0; i < N_pyr*N_steer; i++) {
    free(filters.steered[i]);
    if ( !option ) // synthesis case
      free(filters.mask[i]);
  }
  for(i = 0; i < 1+N_pyr; i++)
    free(filters.lowpass0[i]);
  free(filters.lowpass0);
  free(filters.steered);
  if ( !option ) // synthesis case
    free(filters.mask);
  free(filters.size);
}
