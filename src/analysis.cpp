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
#include <string.h>
#include <math.h>

#include "constraints.h"
#include "mt19937ar.h"
#include "pca.h"
#include "filters.h"
#include "toolbox.h"
#include "ps_lib.h"
#include "pyramid.h"

// Computations of the statistics of an image (from its pyramid decomposition)
// This corresponds to Line 2 of Algorithm 5
// See Appendix B.2 for the color case
// Note that some statistics are computed outside this function
// We use the following variable names:
// - Same scale --> cousins
// - Coarser scale --> parents
static void compute_stats(statsStruct *stats, pyramidStruct pyramid,
                          const imageStruct sample, const filtersStruct filters,
                          const paramsStruct params, int nz)
{
  // parameters
  int N_steer = params.N_steer;
  int N_pyr = params.N_pyr;
  int Na = params.Na;
  int hNa = (Na-1)/2; // variance location in auto-correlation matrix

  // variable declaration
  float meani, vari, theta;
  int i, j, k, l, ind, sx, sy, N_data;

  // image size
  int nx = filters.size[0];
  int ny = filters.size[1];

  // memory allocation
  float **magSteered = (float **) malloc(N_steer*sizeof(float *));
  float **realSteered = (float **) malloc(N_steer*sizeof(float *));
  float **parents = (float **) malloc(N_steer*sizeof(float *));
  float **rparents = (float **) malloc(2*N_steer*sizeof(float *));
  for(j = 0; j < N_steer; j++) {
    magSteered[j] = (float *) malloc(nx*ny*nz*sizeof(float));
    realSteered[j] = (float *) malloc(nx*ny*nz*sizeof(float));
    parents[j] = (float *) malloc(nx*ny*nz*sizeof(float));
    rparents[j] = (float *) malloc(nx*ny*nz*sizeof(float));
    rparents[j + N_steer] = (float *) malloc(nx*ny*nz*sizeof(float));
  }
  fftwf_complex *fft_tmp = (fftwf_complex *)
    fftwf_malloc(nx*ny*nz*sizeof(fftwf_complex));
  fftwf_complex *fft_tmp2 = (fftwf_complex *)
    fftwf_malloc(nx*ny*nz*sizeof(fftwf_complex));
  float *tmp = (float *) malloc(nx*ny*nz*sizeof(float));
  // float *variance = (float *) malloc(nz*sizeof(float));

  // compute the pixel statistics of PCA bands
  if ( nz == 3 ) {
    // central auto-correlation
    // summary statistics (viii)
    compute_auto_cor(stats->autoCorPCA, sample.image, pyramid.in_plan,
                     pyramid.out_plan, pyramid.plan[0], pyramid.iplan[0],
                     nx, ny, nz, Na);

    // skewness and kurtosis
    // summary statistics (ix)
    for (l = 0; l < 3; l++) {
      vari = stats->autoCorPCA[hNa+hNa*Na+l*Na*Na];
      stats->pixelStatsPCA[0 + N_PIXELSTATSPCA*l] =
        compute_skewness(sample.image + l*nx*ny, 0.0, vari, nx*ny);
      stats->pixelStatsPCA[1 + N_PIXELSTATSPCA*l] =
        compute_kurtosis(sample.image + l*nx*ny, 0.0, vari, nx*ny);
    }
  }

  // variance of the high-pass
  // summary statistics (i)(b)
  for(l = 0; l < nz; l++)
    stats->varHigh[l] = compute_moment(pyramid.highband + l*nx*ny, 0.0, 2,
                                       nx*ny);

  // variance for skewness and kurtosis computations
  // if ( nz == 3 )
  //   for(l = 0; l < nz; l++)
  //     variance[l] = stats->eigenValuesPCA[l];
  //  else
  //     variance[0] = stats->pixelStats[3];

  // statistics of the low-frequency residual at each scale
  for (i = 0; i < 1+N_pyr; i++) {
    sx = filters.size[2*i];
    sy = filters.size[2*i+1];

    // apply second low-pass
    do_fft_plan_real(pyramid.plan[i], pyramid.out_plan, pyramid.in_plan,
                     fft_tmp, pyramid.lowband[i], sx*sy, nz);
    pointwise_complexfloat_multiplication(fft_tmp, fft_tmp, filters.lowpass0[i],
                                          sx*sy, nz);
    do_ifft_plan_real(pyramid.iplan[i], pyramid.out_plan, pyramid.in_plan,
                      tmp, fft_tmp, sx*sy, nz);

    // compute central auto-correlation
    // summary statistics (ii)
    compute_auto_cor(stats->autoCorLow[i], tmp, pyramid.in_plan,
                     pyramid.out_plan, pyramid.plan[i], pyramid.iplan[i],
                     sx, sy, nz, Na);

    // compute skewness and kurtosis
    // summary statistics (i)(a)
    for (l = 0; l < nz; l++) {
      vari = stats->autoCorLow[i][hNa + hNa*Na + l*Na*Na];
      // uncomment the following lines to do as in PS matlab implementation
      // the condition is useless because the skewness and kurtosis adjustments
      // are not performed during the analysis if the test fails
      // if ( vari*pow(16, i)/variance[l] > 1e-6) {
        stats->skewLow[i + (1+N_pyr)*l] = compute_skewness(tmp + l*sx*sy, 0.0,
                                                           vari, sx*sy);
        stats->kurtLow[i + (1+N_pyr)*l] = compute_kurtosis(tmp + l*sx*sy, 0.0,
                                                           vari, sx*sy);
      // }
      // else {
      //   stats->skewLow[i + (1+N_pyr)*l] = 0.0;
      //   stats->kurtLow[i + (1+N_pyr)*l] = 3.0;
      // }
    }
  }

  // loop over the scales to compute the statistics of the steered bands
  for(i = 0; i < N_pyr; i++) {
    // sizes
    sx = filters.size[2*i];
    sy = filters.size[2*i+1];

    // loop on the orientation
    for(j = 0; j < N_steer; j++) {
      // index of the corresponding steered band
      ind = j + i*N_steer;

      // compute the magnitude and the real part
      for(k = 0; k < sx*sy*nz; k++) {
        magSteered[j][k] = hypot(pyramid.steered[ind][k][0],
                                 pyramid.steered[ind][k][1]);
        realSteered[j][k] = pyramid.steered[ind][k][0];
      }

      // compute the mean of the magnitude, store it and remove it
      for(l = 0; l < nz; l++) {
        meani = stats->magMeans[ind + (N_pyr*N_steer)*l] =
          mean(magSteered[j] + l*sx*sy, sx*sy);
        for(k = 0; k < sx*sy; k++)
          magSteered[j][k + l*sx*sy] -= meani;
      }
    }

    // compute the central auto-correlation of the modulus
    // summary statistics (iii)
    for(j = 0; j < N_steer; j++) {
      ind = j + i*N_steer;
      compute_auto_cor(stats->autoCorMag[ind], magSteered[j], pyramid.in_plan,
                       pyramid.out_plan, pyramid.plan[i], pyramid.iplan[i],
                       sx, sy, nz, Na);
    }

    // compute the parents (coarser scale) for the cross-correlation
    if ( i == N_pyr - 1 ) { // last scale
      if ( nz == 3) {
        // zoom of the last low-band
        do_fft_plan_real(pyramid.plan[i+1], pyramid.out_plan, pyramid.in_plan,
                         fft_tmp, pyramid.lowband[i+1], 0.25*sx*sy, nz);
        upsampling(fft_tmp2, fft_tmp, sx/2, sy/2, nz);
        do_ifft_plan_real(pyramid.iplan[i], pyramid.out_plan, pyramid.in_plan,
                          tmp, fft_tmp2, sx*sy, nz);

        // rparents are filled shifted version of the zoomed low-band (5 columns)
        // the parents are not used
        memcpy(rparents[0], tmp, nz*sx*sy*sizeof(float));
        shift(rparents[1], tmp, 0, 2, sx, sy, nz);
        shift(rparents[2], tmp, 0, -2, sx, sy, nz);
        shift(rparents[3], tmp, 2, 0, sx, sy, nz);
        shift(rparents[4], tmp, -2, 0, sx, sy, nz);
      }
    }
    else { // not the last scale
      // loop on the orientation
      for(j = 0; j < N_steer; j++) {
        // index of the steered band from the coarser scale
        ind = j + (i+1)*N_steer;

        // zoom
        do_fft_plan(pyramid.plan[i+1], pyramid.out_plan, pyramid.in_plan,
                    fft_tmp, pyramid.steered[ind], 0.25*sx*sy, nz);
        upsampling(fft_tmp2, fft_tmp, sx/2, sy/2, nz);
        do_ifft_plan(pyramid.iplan[i], pyramid.out_plan, pyramid.in_plan,
                          fft_tmp, fft_tmp2, sx*sy, nz);

        // loop over the pixels
        for(k = 0; k < sx*sy*nz; k++) {
          // store the modulus in parents[j]
          parents[j][k] = hypot(fft_tmp[k][0], fft_tmp[k][1]);

          // double the phase of the parents
          theta = 2*atan2(fft_tmp[k][1], fft_tmp[k][0]);

          // store the real part in rparents[j]
          rparents[j][k] = parents[j][k]*cos(theta);

          // store the imaginary part in rparents[j+N_steer]
          rparents[j + N_steer][k] = parents[j][k]*sin(theta);
        }

        // remove the mean of parents[j]
        for(l = 0; l < nz; l++) {
            meani = mean(parents[j] + l*sx*sy, sx*sy);
            for(k = 0; k < sx*sy; k++)
                parents[j][k + l*sx*sy] -= meani;
        }
      }
    }

    // compute the pairwise cross-correlation of the magnitude
    // summary statistics (iv)
    compute_cross_cor(stats->cousinMagCor[i], magSteered, N_steer, sx*sy, nz);

    // compute the cross-correlation of the magnitude with the coarser scale
    // summmary statistics (v)
    if (i < N_pyr - 1)
      compute_cross_scale_cor(stats->parentMagCor[i], magSteered, parents,
                              N_steer, N_steer, sx*sy, nz);

    // compute the pairwise cross-correlation of the real part
    if (nz == 3) {
      // compute the cross-correlation
      // summary statistics (x)
      compute_cross_cor(stats->cousinRealCor[i], realSteered, N_steer, sx*sy, nz);

      // additionnal computation for the last scale
      if (i == N_pyr - 1) {
        // compute the cross-correlation
        // summary statistics (xi)
        compute_cross_cor(stats->cousinRealCor[i+1], rparents, N_SMALLEST, sx*sy, nz);
      }
    }

    // compute cross-correlation of the real part with the coarser scale
    // summary statistics (vi) (and (xii) for the last scale if color)
    if (i < N_pyr - 1 || nz == 3) { // except for grayscale and last scale
      // different size at the coarsest scale
      N_data = (i == N_pyr - 1) ? N_SMALLEST : 2*N_steer;

      // compute the cross-correlation
      compute_cross_scale_cor(stats->parentRealCor[i], realSteered,
                              rparents, N_steer, N_data, sx*sy, nz);
    }
  }

  // free memory
  for(j = 0; j < N_steer; j++) {
    free(magSteered[j]);
    free(realSteered[j]);
    free(parents[j]);
    free(rparents[j]);
    free(rparents[j + N_steer]);
  }
  free(magSteered);
  free(realSteered);
  free(parents);
  free(rparents);
  fftwf_free(fft_tmp);
  fftwf_free(fft_tmp2);
  free(tmp);
  //free(variance);
}

// Analysis of an image
// This corresponds to Line 1 and Line 2 of Algorithm 5
void analysis(statsStruct *stats, imageStruct sample, const paramsStruct params)
{
  int i, l;

  // parameters
  int N_steer = params.N_steer;
  int N_pyr = params.N_pyr;
  int verbose = params.verbose;
  int nx = sample.nx;
  int ny = sample.ny;
  int nz = sample.nz;
  int N = nx*ny;

  // option for the filters and the pyramid building
  int option = 1;

  // compute pixel stats before applying the PCA or adding noise
  // summary statistics (i)(c)
  float m0, var0;
  for (l = 0; l < nz; l++) {
    min_and_max(&stats->pixelStats[0 + N_PIXELSTATS*l],
                &stats->pixelStats[1 + N_PIXELSTATS*l],
                sample.image + l*N, N);
    m0 = stats->pixelStats[2 + N_PIXELSTATS*l] = mean(sample.image + l*N, N);
    var0 = stats->pixelStats[3 + N_PIXELSTATS*l]
      = compute_moment(sample.image + l*N, m0, 2, N);
    stats->pixelStats[4 + N_PIXELSTATS*l] = compute_skewness(sample.image + l*N,
                                                             m0, var0, N);
    stats->pixelStats[5 + N_PIXELSTATS*l] = compute_kurtosis(sample.image + l*N,
                                                             m0, var0, N);
  }

  // add noise to the sample to avoid instability in case of synthetic texture
  if ( nz == 1 ) {
    float factor;
    for (l = 0; l < nz; l++) {
      factor = (stats->pixelStats[1 + N_PIXELSTATS*l]
        - stats->pixelStats[0 + N_PIXELSTATS*l])/100000;
      for (i = 0; i < N; i++)
        sample.image[i + l*N] += factor*mt_genrand_res53();
    }
  }

  // apply PCA (see Appendix B.1)
  if( nz == 3 ) {
    // substract the mean value of each channel (already computed)
    for(l = 0; l < 3; l++)
      for(i = 0; i < N; i++)
        sample.image[i + l*N] -= stats->pixelStats[2 + N_PIXELSTATS*l];

    // compute the color covariance matrix
    // summary statistics (vii)
    compute_covariance(sample.image, stats->covariancePCA, N);

    // compute the change of basis matrix
    eigen_decomposition(stats->covariancePCA, stats->eigenVectorsPCA,
                        stats->eigenValuesPCA);

    // apply PCA
    apply_pca(sample.image, sample.image, stats->eigenVectorsPCA,
              stats->eigenValuesPCA, N);
  }

  // compute the filters and their sizes (for the sample)
  if ( verbose )
    printf("Creating filters for the sample\n");
  filtersStruct filters;
  compute_filters(&filters, nx, ny, N_pyr, N_steer, option);

  // memory allocation for the pyramid
  pyramidStruct pyramid;
  allocate_pyramid(&pyramid, N_pyr, N_steer, nx, ny, nz, option);

  // precomputing the plan for the fft
  precompute_plan(pyramid.plan, pyramid.iplan, pyramid.in_plan,
                  pyramid.out_plan, nx, ny, N_pyr);

  // creating the pyramid for the sample
  if ( verbose )
    printf("Creating the pyramid for the sample\n");
  create_pyramid(pyramid, sample, filters, params, option);

  // computing the statistics of the sample (Line 2 of Algorithm 5)
  if ( verbose )
    printf("Computing the statistics of the sample\n");
  compute_stats(stats, pyramid, sample, filters, params, nz);

  // free memory
  free_filters(filters, N_pyr, N_steer, option);
  free_pyramid(pyramid, N_pyr, N_steer, option);
}

// Analysis of an image
// This corresponds to Line 1 and Line 2 of Algorithm 5
// This function is similar to the analysis() function (above) but it uses
// pre-computed filters and a pre-allocated pyramid
// In addition the noise is not added and there is no verbose mode
void analysis2(statsStruct *stats, imageStruct sample, pyramidStruct pyramid,
               const filtersStruct filters, const paramsStruct params)
{
  int i, l;

  // parameters
  int nx = sample.nx;
  int ny = sample.ny;
  int nz = sample.nz;
  int N = nx*ny;

  // option for the filters and the pyramid building
  int option = 1;

  // compute pixel stats before applying the PCA or adding noise
  // summary statistics (i)(c)
  float m0, var0;
  for (l = 0; l < nz; l++) {
    min_and_max(&stats->pixelStats[0 + N_PIXELSTATS*l],
                &stats->pixelStats[1 + N_PIXELSTATS*l],
                sample.image + l*N, N);
    m0 = stats->pixelStats[2 + N_PIXELSTATS*l] = mean(sample.image + l*N, N);
    var0 = stats->pixelStats[3 + N_PIXELSTATS*l]
      = compute_moment(sample.image + l*N, m0, 2, N);
    stats->pixelStats[4 + N_PIXELSTATS*l] = compute_skewness(sample.image + l*N,
                                                             m0, var0, N);
    stats->pixelStats[5 + N_PIXELSTATS*l] = compute_kurtosis(sample.image + l*N,
                                                             m0, var0, N);
  }

  // apply PCA (see Appendix B.1)
  if( nz == 3 ) {
    // substract the mean value of each channel (already computed)
    for(l = 0; l < 3; l++)
      for(i = 0; i < N; i++)
        sample.image[i + l*N] -= stats->pixelStats[2 + N_PIXELSTATS*l];

    // compute the color covariance matrix
    // summary statistics (vii)
    compute_covariance(sample.image, stats->covariancePCA, N);

    // compute the change of basis matrix
    eigen_decomposition(stats->covariancePCA, stats->eigenVectorsPCA,
                        stats->eigenValuesPCA);

    // apply PCA
    apply_pca(sample.image, sample.image, stats->eigenVectorsPCA,
              stats->eigenValuesPCA, N);
  }

  // creating the pyramid for the sample
  create_pyramid(pyramid, sample, filters, params, option);

  // computing the statistics of the sample (Line 2 of Algorithm 5)
  compute_stats(stats, pyramid, sample, filters, params, nz);
}
