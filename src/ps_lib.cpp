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
#ifdef _OPENMP
#include <omp.h>
#endif

#include "analysis.h"
#include "constraints.h"
#include "pca.h"
#include "periodic_plus_smooth.h"
#include "ps_lib.h"
#include "synthesis.h"
#include "zoom_bilinear.h"

// Compute the weighted average of two arrays
static void averaging(float *list, float *list2, int N, float weight)
{
  for(int i = 0; i<N; i++)
    list[i] = weight*list[i] + (1 - weight)*list2[i];
}

// Interpolate the statistics of two textures
// This is done by a weighted average of the statistics
// For color images the eigen decomposition has to be recomputed
static void interpolate_stats(statsStruct *stats, statsStruct *stats2,
                              const paramsStruct params, int nz)
{
  int N_pyr = params.N_pyr;
  int N_steer = params.N_steer;
  int Na = params.Na;
  float w = params.interpWeight;
  int i;

  // weighted averaging of the statistics
  averaging(stats->pixelStats, stats2->pixelStats, N_PIXELSTATS*nz, w);
  averaging(stats->skewLow, stats2->skewLow, (1+N_pyr)*nz, w);
  averaging(stats->kurtLow, stats2->kurtLow, (1+N_pyr)*nz, w);
  averaging(stats->varHigh, stats2->varHigh, nz, w);
  averaging(stats->magMeans, stats2->magMeans, N_pyr*N_steer*nz, w);
  for(i = 0; i < 1+N_pyr; i++)
    averaging(stats->autoCorLow[i],stats2->autoCorLow[i], Na*Na*nz, w);
  for(i = 0; i < N_pyr*N_steer; i++)
    averaging(stats->autoCorMag[i],stats2->autoCorMag[i], Na*Na*nz, w);
  for(i = 0; i < N_pyr; i++)
    averaging(stats->cousinMagCor[i], stats2->cousinMagCor[i],
              N_steer*N_steer*nz*nz, w);
  for(i = 0; i < N_pyr-1; i++)
    averaging(stats->parentMagCor[i], stats2->parentMagCor[i],
              N_steer*N_steer*nz*nz, w);
  for(i = 0; i < N_pyr-1; i++)
    averaging(stats->parentRealCor[i], stats2->parentRealCor[i],
              2*N_steer*N_steer*nz*nz, w);
  if ( nz == 3 ) {
    averaging(stats->parentRealCor[N_pyr-1], stats2->parentRealCor[N_pyr-1],
              N_SMALLEST*N_steer*nz*nz, w);
    averaging(stats->pixelStatsPCA, stats2->pixelStatsPCA,
              N_PIXELSTATSPCA*nz, w);
    for(i = 0; i < N_pyr; i++)
      averaging(stats->cousinRealCor[i], stats2->cousinRealCor[i],
                N_steer*N_steer*nz*nz, w);
    averaging(stats->cousinRealCor[N_pyr], stats2->cousinRealCor[N_pyr],
                N_SMALLEST*N_SMALLEST*nz*nz, w);
    averaging(stats->autoCorPCA, stats2->autoCorPCA, Na*Na*nz, w);
    for(i = 0; i < 3; i++)
      averaging(stats->covariancePCA[i], stats2->covariancePCA[i], 3, w);

    // recompute the eigen decomposition
    eigen_decomposition(stats->covariancePCA, stats->eigenVectorsPCA,
                        stats->eigenValuesPCA);
  }
}

// Main function for computing the texture from the sample (Algorithm 5)
void ps(imageStruct texture, imageStruct sample, imageStruct sample2,
        paramsStruct params)
{
  // start threaded fftw if FFTW_NTHREADS is defined
  #ifdef FFTW_NTHREADS
    fftwf_init_threads();
    #ifdef _OPENMP
      fftwf_plan_with_nthreads(omp_get_max_threads());
    #endif
  #endif

  // define parameters
  int nxin = sample.nx, nyin = sample.ny;
  int nxin2 = sample2.nx, nyin2 = sample2.ny;
  int nxout = texture.nx, nyout = texture.ny;
  int nz = sample.nz;
  int verbose = params.verbose;
  float interpWeight = params.interpWeight;

  // compute the periodic plus smooth decomposition
  float *smooth = NULL, *smooth2 = NULL;
  if ( params.edge_handling ) {
    if ( verbose )
      printf("Periodic plus smooth decomposition of the sample\n");
    smooth = (float *) malloc(nxin*nyin*nz*sizeof(float));
    periodic_plus_smooth_decomposition(sample.image, smooth, sample.image,
                                       nxin, nyin, nz);
    if ( interpWeight >= 0 ) {
      smooth2 = (float *) malloc(nxin2*nyin2*nz*sizeof(float));
      periodic_plus_smooth_decomposition(sample2.image, smooth2,
                                         sample2.image, nxin2, nyin2, nz);
    }
  }

  // allocate memory for the statistics of the sample
  statsStruct stats;
  allocate_stats(&stats, params, nz);

  // analysis of the sample (Line 1 and Line 2)
  if ( verbose )
    printf("Analyzing the sample\n");
  analysis(&stats, sample, params);

  // texture interpolation case: analysis of the second sample
  if ( interpWeight >= 0 ) {
    // allocate memory for the statistics of the second sample
    statsStruct stats2;
    allocate_stats(&stats2, params, nz);

    // analysis of the second sample
    if ( verbose )
      printf("Analyzing the second sample\n");
    analysis(&stats2, sample2, params);

    // interpolation of the statistics
    interpolate_stats(&stats, &stats2, params, nz);

    // free memory
    free_stats(stats2, params, nz);
  }

  // write the constraints in a file
  if ( params.statistics )
    write_statistics(stats, params, nz);

  // synthesis of the texture (Line 3 to Line 9)
  if ( verbose )
    printf("Synthesizing the texture\n");
  synthesis(texture, stats, params);

  // add the smooth component (possibly zoomed)
  if ( params.add_smooth ) {
    if ( verbose )
      printf("Adding the smooth component\n");
    float *smooth_zoomed = (float *) malloc(nxout*nyout*nz*sizeof(float));
    zoom_bilinear(smooth_zoomed, nxout, nyout, smooth, nxin, nyin, nz);
    if ( interpWeight >= 0 ) {
      float *smooth2_zoomed = (float *) malloc(nxout*nyout*nz*sizeof(float));
      zoom_bilinear(smooth2_zoomed, nxout, nyout, smooth2, nxin2, nyin2, nz);
      averaging(smooth_zoomed, smooth2_zoomed, nxout*nyout*nz, interpWeight);
      free(smooth2_zoomed);
    }
    for(int i = 0; i < nxout*nyout*nz; i++)
      texture.image[i] += smooth_zoomed[i];
    free(smooth_zoomed);
  }

  // free memory
  free_stats(stats, params, nz);
  if ( params.edge_handling ) {
    free(smooth);
    if ( interpWeight >= 0)
      free(smooth2);
  }

  // fftw cleanup
  fftwf_cleanup();
  #ifdef FFTW_NTHREADS
    fftwf_cleanup_threads();
  #endif
}
