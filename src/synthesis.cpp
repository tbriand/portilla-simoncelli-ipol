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

#include "analysis.h"
#include "constraints.h"
#include "filters.h"
#include "mt19937ar.h"
#include "pca.h"
#include "ps_lib.h"
#include "pyramid.h"
#include "toolbox.h"

/* M_PI is a POSIX definition */
#ifndef M_PI
/** macro definition for Pi */
#define M_PI 3.14159265358979323846
#endif                          /* !M_PI */

// Adjust the summary statistics of an image (Algorithm 4)
// See Appendix B.2 for the color case
// The filtering are performed in the Fourier domain (see Section 2.3)
// Computations are saved by avoiding the unnecessary DFT and iDFT computations
// between successive steps
// We use the following variable names:
// - Same scale    <--> cousins
// - Coarser scale <--> parents
// We recall that:
// - cmask[0] <--> adjust marginal statistics (1) or not (0)
// - cmask[1] <--> adjust auto-correlation (1) or not (0)
// - cmask[2] <--> adjust magnitude correlation (1) or not (0)
// - cmask[3] <--> adjust real correlation (1) or not (0)
static void adjust_constraints(imageStruct texture, statsStruct stats,
                               pyramidStruct pyramid,
                               const filtersStruct filters,
                               const paramsStruct params)
{
  // parameters
  int N_steer = params.N_steer;
  int N_pyr = params.N_pyr;
  int Na = params.Na;
  int hNa = (Na-1)/2;
  int cmask[4];
  memcpy(cmask, params.cmask, 4*sizeof(int));
  int nx = texture.nx;
  int ny = texture.ny;
  int nz = texture.nz;

  int i, j, k, l, ind, N_data;
  float meani, vari, theta, factor;

  // allocate memory
  fftwf_complex *fft_tmp = (fftwf_complex *)
    fftwf_malloc( nx*ny*nz*sizeof(fftwf_complex));
  fftwf_complex *fft_tmp2 = (fftwf_complex *)
    fftwf_malloc( nx*ny*nz*sizeof(fftwf_complex));
  fftwf_complex *fft_tmp3 = (fftwf_complex *)
    fftwf_malloc( nx*ny*nz*sizeof(fftwf_complex));
  float **tmpSteered = (float **) malloc(N_steer*sizeof(float *));
  float **parents = (float **) malloc(N_steer*sizeof(float *));
  float **rparents = (float **) malloc(2*N_steer*sizeof(float *));
  for(j = 0; j < N_steer; j++) {
    tmpSteered[j] = (float *) malloc(nx*ny*nz* sizeof(float));
    parents[j] = (float *) malloc(nx*ny*nz* sizeof(float));
    rparents[j] = (float *) malloc(nx*ny*nz* sizeof(float));
    rparents[j + N_steer] = (float *) malloc(nx*ny*nz* sizeof(float));
  }
  float *cor_tmp = (float *) malloc(2*N_steer*nz*sizeof(float));
  float *variance = (float *) malloc(nz*sizeof(float));
  float *variance2 = (float *) malloc(nz*sizeof(float));

  // variance for auto-correlation, skewness and kurtosis adjustments
  if ( nz == 3 )
    for(l = 0; l < nz; l++)
      variance[l] = stats.eigenValuesPCA[l];
  else
    variance[0] = stats.pixelStats[3];

  // tolerance for variance ratio in skewness and kurtosis adjustments
  float tol = (nz == 3) ? 1e-3 : 1e-4;

  // matching of the first low-band
  int sx = filters.size[2*N_pyr];
  int sy = filters.size[2*N_pyr+1];

  // adjust spatial-color cross-correlation in the low-pass residual
  // see Section B.2
  if ( nz == 3 ) {
    // compute the fft of the low-band
    do_fft_plan_real(pyramid.plan[N_pyr], pyramid.out_plan, pyramid.in_plan,
                     fft_tmp, pyramid.lowband[N_pyr], sx*sy, nz);

    // up-sampling in the Fourier domain
    upsampling(fft_tmp2, fft_tmp, sx, sy, nz);

    // texture.image is temporary used as a buffer
    do_ifft_plan_real(pyramid.iplan[N_pyr - 1], pyramid.out_plan,
                      pyramid.in_plan, texture.image, fft_tmp2, 4*sx*sy, nz);

    // rparents are filled with shifted version of the zoomed low-band
    memcpy(rparents[0], texture.image, nz*2*sx*2*sy*sizeof(float));
    shift(rparents[1], texture.image, 0, 2, 2*sx, 2*sy, nz);
    shift(rparents[2], texture.image, 0, -2, 2*sx, 2*sy, nz);
    shift(rparents[3], texture.image, 2, 0, 2*sx, 2*sy, nz);
    shift(rparents[4], texture.image, -2, 0, 2*sx, 2*sy, nz);

    // adjust cross-correlation
    adjust_cross_cor(rparents, stats.cousinRealCor[N_pyr], N_SMALLEST, 2*sx*2*sy, nz);

    // average over the N_SMALLEST=5 bands after unshifting
    // use rparents[5] as buffer for unshifting
    for(i = 0; i < 4*sx*sy*nz; i++)
      texture.image[i] = rparents[0][i];

    shift(rparents[5], rparents[1], 0, -2, 2*sx, 2*sy, nz);
    for(i = 0; i < 4*sx*sy*nz; i++)
      texture.image[i] += rparents[5][i];

    shift(rparents[5], rparents[2], 0, 2, 2*sx, 2*sy, nz);
    for(i = 0; i < 4*sx*sy*nz; i++)
      texture.image[i] += rparents[5][i];

    shift(rparents[5], rparents[3], -2, 0, 2*sx, 2*sy, nz);
    for(i = 0; i < 4*sx*sy*nz; i++)
      texture.image[i] += rparents[5][i];

    shift(rparents[5], rparents[4], 2, 0, 2*sx, 2*sy, nz);
    for(i = 0; i < 4*sx*sy*nz; i++) {
      texture.image[i] += rparents[5][i];
      texture.image[i] /= N_SMALLEST;
    }

    // compute the fft
    do_fft_plan_real(pyramid.plan[N_pyr - 1], pyramid.out_plan,
                     pyramid.in_plan, fft_tmp, texture.image, 4*sx*sy, nz);

    // down-sampling in the Fourier domain
    downsampling(fft_tmp3, fft_tmp, sx, sy, nz);
  }
  else { // for nz = 1 just compute the fft of the low-band in fft_tmp3
    do_fft_plan_real(pyramid.plan[N_pyr], pyramid.out_plan, pyramid.in_plan,
                     fft_tmp3, pyramid.lowband[N_pyr], sx*sy, nz);
  }

  // apply the second low-pass to the low-band (without modifying pyramid.lowband[N_pyr])
  // This corresponds to Line 2
  // the fft is already computed in fft_tmp3 (we do not modify it for rparents below)
  // texture.image is temporary used as a buffer
  pointwise_complexfloat_multiplication(fft_tmp2, fft_tmp3,
                                        filters.lowpass0[N_pyr], sx*sy, nz);
  do_ifft_plan_real(pyramid.iplan[N_pyr], pyramid.out_plan, pyramid.in_plan,
                    texture.image, fft_tmp2, sx*sy, nz);

  // set the variance value for testing
  for (l = 0; l < nz; l++)
    variance2[l] = stats.autoCorLow[N_pyr][hNa+hNa*Na+l*Na*Na]*pow(16, N_pyr);

  // adjust auto-correlation (Line 3)
  if ( cmask[1] )
    adjust_auto_cor(texture.image, stats.autoCorLow[N_pyr], variance,
                 variance2, pyramid.in_plan, pyramid.out_plan,
                 pyramid.plan[N_pyr], pyramid.iplan[N_pyr],
                 sx, sy, nz, Na, N_pyr);

  // adjust skewness and kurtosis (Line 4 and Line 5)
  if ( cmask[0] ) {
    for (l = 0; l < nz; l++) {
      if ( variance2[l]/variance[l] > tol) {
        adjust_skewness(texture.image + l*sx*sy,
                        stats.skewLow[N_pyr + (1+N_pyr)*l], sx*sy);
        adjust_kurtosis(texture.image + l*sx*sy,
                        stats.kurtLow[N_pyr + (1+N_pyr)*l], sx*sy);
      }
    }
  }

  // compute the fft of the lowest band
  do_fft_plan_real(pyramid.plan[N_pyr], pyramid.out_plan, pyramid.in_plan,
                   fft_tmp, texture.image, sx*sy, nz);

  // Loop over the scale (Line 6)
  for(i = 0; i < N_pyr; i++) {
    // up-sampling of the low-band in Fourier domain (Line 7)
    upsampling(fft_tmp2, fft_tmp, sx, sy, nz);

    // size after zoom
    sx *= 2;
    sy *= 2;

    // compute the real and magnitude parents
    // the update of parents[j] for i>0 is the second part of Line 9
    if ( cmask[2] || cmask[3] ) {
      if (i == 0 && nz == 3) { // lowest scale for color
        // up-sample the last low-band (whose fft is already in fft_tmp3)
        // texture.image is temporary used as a buffer
        upsampling(fft_tmp, fft_tmp3, sx/2, sy/2, nz);
        do_ifft_plan_real(pyramid.iplan[N_pyr - 1], pyramid.out_plan,
                          pyramid.in_plan, texture.image, fft_tmp,
                          sx*sy, nz);

        // rparents are filled shifted version of the zoomed low-band (5 columns)
        // the low-band version used does not have its auto-correlation,
        // skewness and kurtosis adjusted
        // FYI: parents remains empty
        memcpy(rparents[0], texture.image, nz*sx*sy*sizeof(float));
        shift(rparents[1], texture.image, 0, 2, sx, sy, nz);
        shift(rparents[2], texture.image, 0, -2, sx, sy, nz);
        shift(rparents[3], texture.image, 2, 0, sx, sy, nz);
        shift(rparents[4], texture.image, -2, 0, sx, sy, nz);
      }
      else if (i > 0) { // not the last scale
        // loop on the orientation
        for(j = 0; j < N_steer; j++) {
          // index of the steered band from the coarser scale
          ind = N_steer*(N_pyr - i) + j;

          // zoom
          do_fft_plan(pyramid.plan[N_pyr-i], pyramid.out_plan,
                      pyramid.in_plan, fft_tmp, pyramid.steered[ind],
                      0.25*sx*sy, nz);
          upsampling(fft_tmp3, fft_tmp, sx/2, sy/2, nz);
          do_ifft_plan(pyramid.iplan[N_pyr-1-i], pyramid.out_plan,
                       pyramid.in_plan, fft_tmp, fft_tmp3, sx*sy, nz);

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
    }

    // adjust cross-correlation with magnitudes at other orientations/scales
    if( cmask[2] ) {
      // loop on the orientation to get the cousins (Line 8)
      // NOTE: cousins <--> tmpSteered
      for(j = 0; j < N_steer; j++) {
        // index of the corresponding steered band
        ind = N_steer*(N_pyr - i - 1) + j;

        // compute the magnitude of the cousins
        for(k = 0; k < sx*sy*nz; k++)
          tmpSteered[j][k] = hypot(pyramid.steered[ind][k][0],
                                   pyramid.steered[ind][k][1]);

        // compute the mean of the magnitude and remove it (first part of Line 9)
        for(l = 0; l < nz; l++) {
          meani = mean(tmpSteered[j] + l*sx*sy, sx*sy);
          for(k = 0; k < sx*sy; k++)
            tmpSteered[j][k + l*sx*sy] -= meani;
        }
      }

      // adjust the cross-correlations of the magnitudes (Line 10)
      if( i == 0 ) { // last scale
        // adjust the pairwise cross-correlation
        adjust_cross_cor(tmpSteered, stats.cousinMagCor[N_pyr-1-i],
                      N_steer, sx*sy, nz);
      }
      else {
        // adjust cross-correlation with cousins and parents
        adjust_cross_scale_cor(tmpSteered, parents,
                            stats.cousinMagCor[N_pyr-1-i],
                            stats.parentMagCor[N_pyr-1-i], N_steer,
                            N_steer, sx*sy, nz);
      }

      // loop over the orientation (Line 11)
      for(j = 0; j < N_steer; j++) {
        ind = N_steer*(N_pyr - i - 1) + j;

        // adjust auto-correlation (Line 12)
        if ( cmask[1] )
          adjust_auto_cor(tmpSteered[j], stats.autoCorMag[ind], variance,
                       variance, pyramid.in_plan, pyramid.out_plan,
                       pyramid.plan[N_pyr-1-i], pyramid.iplan[N_pyr-1-i],
                       sx, sy, nz, Na, N_pyr-1-i);

        // adjust the mean (Line 13)
        for(l = 0; l < nz; l++)
          for(k = 0; k < sx*sy; k++)
            tmpSteered[j][k + l*sx*sy] +=
              stats.magMeans[ind + (N_pyr*N_steer)*l];

        // mag0 is computed to have an idea of the order of magnitude for the test
        float mag0 = 0.0;
        for(l = 0; l < nz; l++)
          mag0 += stats.magMeans[ind + (N_pyr*N_steer)*l];
        mag0 /= nz;

        // adjust magnitude (Line 13 and 14)
        float mag;
        for(k = 0; k < nz*sx*sy; k++) {
          // take positive part since a magnitude has to be non-negative (Line 13)
          tmpSteered[j][k] = (tmpSteered[j][k] < 0 ) ? 0 : tmpSteered[j][k];

          // compute original magnitude
          mag = hypot(pyramid.steered[ind][k][0], pyramid.steered[ind][k][1]);
          mag = (mag < (1e-4)*mag0) ? 1 : mag;

          // impose new magnitude (Line 14)
          factor = tmpSteered[j][k]/mag;
          tmpSteered[j][k] = pyramid.steered[ind][k][0]*factor;
        }
      }
    }
    else { // put real cousins in tmpSteered
      for(j = 0; j < N_steer; j++) {
        ind = N_steer*(N_pyr - i - 1) + j;
        for(k = 0; k < nz*sx*sy; k++)
          tmpSteered[j][k] = pyramid.steered[ind][k][0];
      }
    }

    // adjust cross-correlation of real parts at other orientations/scales (Line 15 to Line 17)
    if ( cmask[3] ) {
      if( nz == 3 && cmask[1] ) {
        N_data = (i == 0) ? N_SMALLEST : 2*N_steer;
        adjust_cross_scale_cor(tmpSteered, rparents,
                               stats.cousinRealCor[N_pyr - 1 - i],
                               stats.parentRealCor[N_pyr - 1 - i],
                               N_steer, N_data, sx*sy, nz);
      }
      else if ( i > 0 ) {
        // adjust correlation with rcousins and rparents (rcousins, Cr0, rparents, Crx0)
        if ( nz == 1 ) { // case nz == 1 and i > 0 (Line 17)
          for(j = 0; j < N_steer; j++) {
            vari = compute_moment(tmpSteered[j], 0, 2, sx*sy);
            for(k = 0; k < 2*N_steer; k++)
              cor_tmp[k] = stats.parentRealCor[N_pyr-1-i][k + j*2*N_steer];
            adjust_cross_scale_cor(tmpSteered + j, rparents, &vari, cor_tmp, 1,
                                2*N_steer, sx*sy, 1);
          }
        }
        else { // case (nz == 3 and cmask[1] == 0 and i > 0)
          for(j = 0; j < N_steer; j++)
            for(l = 0; l < nz; l++) {
              ind = j + l*N_steer;
              vari = stats.cousinRealCor[N_pyr - 1 - i][ind + ind*N_steer*nz];
              for(k = 0; k < 2*N_steer*nz; k++)
                cor_tmp[k] = stats.parentRealCor[N_pyr-1-i][k + ind*2*N_steer*nz];
              adjust_cross_scale_cor2(tmpSteered[j] + l*sx*sy, rparents, vari,
                                   cor_tmp, 2*N_steer, sx*sy);
            }
        }
      }
    }

    // re-create low-band (Line 18 to Line 20)
    for(j = 0; j < N_steer; j++) {
      ind = N_steer*(N_pyr - i - 1) + j;

      // impose real part and take real part
      for(k = 0; k < sx*sy*nz; k++) {
        fft_tmp3[k][0] = tmpSteered[j][k];
        fft_tmp3[k][1] = 0.0;
      }

      // apply mask to get analytic signal (Line 19)
      do_fft_plan(pyramid.plan[N_pyr-1-i], pyramid.out_plan, pyramid.in_plan,
                  fft_tmp, fft_tmp3, sx*sy, nz);
      pointwise_complexfloat_multiplication(fft_tmp, fft_tmp, filters.mask[ind],
                                            sx*sy, nz);

      // save the analytic version (for the next scale)
      if ( i < N_pyr - 1) // do not save for the finest scale
        do_ifft_plan(pyramid.iplan[N_pyr-1-i], pyramid.out_plan, pyramid.in_plan,
                     pyramid.steered[ind], fft_tmp, sx*sy, nz);

      // apply steered filter (Line 20)
      // NOTE useless to take the real part of the signal before
      pointwise_complexfloat_multiplication(fft_tmp, fft_tmp,
                                            filters.steered[ind], sx*sy, nz);

      // update buffer fft_tmp2
      // the factor 0.5 is required to compensate the factor 2 in filters.steered
      for(k = 0; k < sx*sy*nz; k++) {
        fft_tmp2[k][0] += 0.5*fft_tmp[k][0];
        fft_tmp2[k][1] += 0.5*fft_tmp[k][1];
      }
    }

    // apply low-pass to the sums of oriented (Line 22)
    pointwise_complexfloat_multiplication(fft_tmp, fft_tmp2,
                                          filters.lowpass0[N_pyr-1-i],
                                          sx*sy, nz);

    // compute the low-band in the spatial domain
    // the real part of Line 21 is implicitly taken here
    do_ifft_plan_real(pyramid.iplan[N_pyr-1-i], pyramid.out_plan,
                      pyramid.in_plan, texture.image, fft_tmp, sx*sy, nz);

    // set the variance value for testing
    for (l = 0; l < nz; l++)
       variance2[l] = stats.autoCorLow[N_pyr-1-i][hNa + hNa*Na + l*Na*Na]
        * pow(16, N_pyr - 1 - i);

    // adjust auto-correlation (Line 23)
    if ( cmask[1] )
      adjust_auto_cor(texture.image, stats.autoCorLow[N_pyr-1-i],
                   variance, variance2, pyramid.in_plan,
                   pyramid.out_plan, pyramid.plan[N_pyr-1-i],
                   pyramid.iplan[N_pyr-1-i], sx, sy, nz, Na, N_pyr - 1 - i);

    // adjust skewness and kurtosis (Line 24 and Line 25)
    if ( cmask[0] ) {
      for (l = 0; l < nz; l++) {
        if ( variance2[l]/variance[l] > tol) {
          adjust_skewness(texture.image + l*sx*sy,
                          stats.skewLow[N_pyr-1-i + (1+N_pyr)*l], sx*sy);
          adjust_kurtosis(texture.image + l*sx*sy,
                          stats.kurtLow[N_pyr-1-i + (1+N_pyr)*l], sx*sy);
        }
      }
    }

    // compute fft except at the finest scale
    if( i < N_pyr - 1 )
      do_fft_plan_real(pyramid.plan[N_pyr-1-i], pyramid.out_plan,
                       pyramid.in_plan, fft_tmp, texture.image, sx*sy, nz);
  }

  // adjust variance in high-pass if higher than desired (Line 26)
  if ( cmask[1] || cmask[2] || cmask[3] ) {
      for(l = 0; l < nz; l++) {
        vari = compute_moment(pyramid.highband + l*nx*ny, 0.0, 2, nx*ny);
        if( vari > stats.varHigh[l]) {
          float factor = sqrt(stats.varHigh[l]/vari);
          for(k = 0; k < nx*ny; k++)
            pyramid.highband[k + l*nx*ny] *= factor;
        }
      }
  }

  // apply the high-pass a second time (Line 27)
  do_fft_plan_real(pyramid.plan[0], pyramid.out_plan, pyramid.in_plan,
                   fft_tmp, pyramid.highband, nx*ny, nz);
  pointwise_complexfloat_multiplication(fft_tmp, fft_tmp, filters.highpass0,
                                        nx*ny, nz);
  do_ifft_plan_real(pyramid.iplan[0], pyramid.out_plan, pyramid.in_plan,
                    pyramid.highband, fft_tmp, nx*ny, nz);

  // sum of low-band and high-band (Line 28)
  for(k = 0; k < nx*ny*nz; k++)
    texture.image[k] += pyramid.highband[k];

  // color handling (see Section B.2)
  if( nz == 3 ) {
    // adjust auto-correlation of PCA bands
    if ( cmask[1] )
      adjust_auto_cor(texture.image, stats.autoCorPCA, variance, variance,
                   pyramid.in_plan, pyramid.out_plan, pyramid.plan[0],
                   pyramid.iplan[0], nx, ny, nz, Na, 0);

    // "pixel" stats of PCA channels
    if ( cmask[0] ) {
      for(l = 0; l < nz; l++) {
        // adjust the variance to 1 and set the mean to 0
        adjust_mean_variance(texture.image + l*nx*ny, 0.0, 1.0, nx*ny);

        // adjust the skewness
        adjust_skewness(texture.image + l*nx*ny,
                        stats.pixelStatsPCA[0 + N_PIXELSTATSPCA*l], nx*ny);

        // adjust the kurtosis
        adjust_kurtosis(texture.image + l*nx*ny,
                        stats.pixelStatsPCA[1 + N_PIXELSTATSPCA*l], nx*ny);
      }
    }

    // adjust the color covariance matrix
    float eye[3][3] =
    {
      {1.0,0.0,0.0},
      {0.0,1.0,0.0},
      {0.0,0.0,1.0}
    };
    adjust_covariance_color(texture.image, eye, nx*ny);

    // apply inverse PCA transform
    apply_inverse_pca(texture.image, texture.image, stats.eigenVectorsPCA,
                      stats.eigenValuesPCA, nx*ny);
  }

  // pixel stats
  for(l = 0; l < nz; l++) {
    if ( cmask[0] ) {
      // adjust the mean and variance (Line 29)
      adjust_mean_variance(texture.image + l*nx*ny, 0.0,
                           stats.pixelStats[3 + l*N_PIXELSTATS], nx*ny);

      // adjust the skewness (Line 30)
      adjust_skewness(texture.image + l*nx*ny,
                      stats.pixelStats[4 + N_PIXELSTATS*l], nx*ny);

      // adjust the kurtosis (Line 31)
      adjust_kurtosis(texture.image + l*nx*ny,
                      stats.pixelStats[5 + N_PIXELSTATS*l], nx*ny);
    }

    // adjust the mean (Line 32)
    meani = stats.pixelStats[2 + l*N_PIXELSTATS];
    for(k = 0; k < nx*ny; k++)
      texture.image[k + l*nx*ny] += meani;

    // adjust the range (Line 33)
    if ( cmask[0] )
      adjust_range(texture.image + l*nx*ny, stats.pixelStats[0 + l*N_PIXELSTATS],
                   stats.pixelStats[1 + l*N_PIXELSTATS], nx*ny);
  }

  // free memory
  fftwf_free(fft_tmp);
  fftwf_free(fft_tmp2);
  fftwf_free(fft_tmp3);
  for(j = 0; j < N_steer; j++) {
    free(tmpSteered[j]);
    free(parents[j]);
    free(rparents[j]);
    free(rparents[j + N_steer]);
  }
  free(tmpSteered);
  free(parents);
  free(rparents);
  free(variance);
  free(variance2);
  free(cor_tmp);
}

// Iterative synthesis of the texture given the summary statistics
// This corresponds to Line 3 to Line 9 of Algorithm 5
void synthesis(imageStruct texture, const statsStruct stats, const paramsStruct params) {
  int i, k, l;

  // parameters
  int N_steer = params.N_steer;
  int N_pyr = params.N_pyr;
  int N_iteration = params.N_iteration;
  int verbose = params.verbose;
  int statistics = params.statistics;
  int nxout = texture.nx;
  int nyout = texture.ny;
  int nz = texture.nz;

  // option for the filters and pyramid building (synthesis case)
  int option1 = 0;
  // option for the pyramid building
  // the analysis case is required for computing the statistics
  int option2 = ( statistics ) ? 1 : 0;

  // compute the filters and their sizes
  if ( verbose )
    printf("Creating filters for the texture\n");
  filtersStruct filters;
  compute_filters(&filters, nxout, nyout, N_pyr, N_steer, option1);

  // memory allocation for the pyramid
  pyramidStruct pyramid;
  allocate_pyramid(&pyramid, N_pyr, N_steer, nxout, nyout, nz, option2);

  // precompute the plans for the FFTs and iFFTs
  precompute_plan(pyramid.plan, pyramid.iplan, pyramid.in_plan, pyramid.out_plan, nxout, nyout, N_pyr);

  // initialize the output texture if the input noise is not provided (Line 3)
  if ( params.noise == 0 ) {
    if( nz == 3) {
      // initialize noise (whatever the variance is)
      for (i = 0; i < 3*nxout*nyout; i++)
        texture.image[i] = sqrt(-2*log(mt_genrand_res53()))
          * cos(2*M_PI*mt_genrand_res53());

      // adjust the mean to 0
      for(l = 0; l < 3; l++) {
        float m = mean(texture.image + l*nxout*nyout, nxout*nyout);
        for(i = 0; i < nxout*nyout; i++)
          texture.image[i + l*nxout*nyout] -= m;
      }

      // adjust the color covariance matrix
      adjust_covariance_color(texture.image, stats.covariancePCA, nxout*nyout);
    }
    else { // nz = 1
      // The mean and the variance are the same as the input ones
      // mt_genrand_res53 generates a uniform noise in [0,1]
      float factor = sqrt(stats.pixelStats[3]); // to have the correct variance
      float noise;
      for (i = 0; i < nxout*nyout; i++) {
        noise = sqrt(-2*log(mt_genrand_res53()))
          * cos(2*M_PI*mt_genrand_res53());
        texture.image[i] = stats.pixelStats[2] + factor*noise;
      }
    }
  }

  // allocate memory and write statistics in a file
  // texture2 is required to avoid modifications of the image in the color case
  statsStruct stats2;
  imageStruct texture2;
  if ( statistics ) {
    // allocate memory
    allocate_stats(&stats2, params, nz);
    texture2.nx = nxout;
    texture2.ny = nyout;
    texture2.nz = nz;
    texture2.image = (float *) malloc( nxout*nyout*nz*sizeof(float) );

    // write statistrics in a file
    memcpy(texture2.image, texture.image, nxout*nyout*nz*sizeof(float));
    analysis2(&stats2, texture2, pyramid, filters, params);
    write_statistics(stats2, params, nz);
  }

  // to increase the convergence rate (Line 4)
  // tmp contains the previous image
  float *tmp = NULL;
  if ( nz == 1 ) {
    tmp = (float *) malloc( nxout*nyout*nz*sizeof(float) );
    memcpy(tmp, texture.image, nxout*nyout*nz*sizeof(float));
  }

  // Loop in order to create the texture (Line 5)
  if ( verbose )
    printf("Iterative synthesis of the texture\n");
  for(k = 0; k < N_iteration; k++) {
    // color handling
    if( nz == 3) {
      // substract the mean value
      for(l = 0; l < 3; l++) {
        float meani = mean(texture.image + l*nxout*nyout, nxout*nyout);
        for(i = 0; i < nxout*nyout; i++)
          texture.image[i + l*nxout*nyout] -= meani;
      }

      // apply direct PCA transform
      apply_pca(texture.image, texture.image, stats.eigenVectorsPCA,
                stats.eigenValuesPCA, nxout*nyout);
    }

    // create the pyramid (Line 6)
    create_pyramid(pyramid, texture, filters, params, option1);

    // adjust the constraints (Line 7)
    adjust_constraints(texture, stats, pyramid, filters, params);

    // write statistics in a file
    if ( statistics ) {
      memcpy(texture2.image, texture.image, nxout*nyout*nz*sizeof(float));
      analysis2(&stats2, texture2, pyramid, filters, params);
      write_statistics(stats2, params, nz);
    }

    // convergence accelerator
    if ( nz == 1 ) {
      float tmp_value;
      if( k < N_iteration - 1 )
        for(i = 0; i < nxout*nyout*nz; i++) {
          // new value
          tmp_value = texture.image[i];

          // accelerator (Line 8)
          texture.image[i] += 0.8*(texture.image[i] - tmp[i]);

          // update old image (Line 9)
          tmp[i] = tmp_value;
        }
    }
  }

  // free memory
  if ( statistics ) {
    free_stats(stats2, params, nz);
    free(texture2.image);
  }
  if ( nz == 1)
    free(tmp);
  free_filters(filters, N_pyr, N_steer, option1);
  free_pyramid(pyramid, N_pyr, N_steer, option2);
}
