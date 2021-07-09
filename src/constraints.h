// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2021, Thibaud Briand <briand.thibaud@gmail.com>
// Copyright (C) 2021, Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
// All rights reserved.

/* constraints.cpp contains the functions for computing and
 * adjusting the statistics (see Appendix A)
 */

#ifndef CONSTRAINTS_H
#define CONSTRAINTS_H

#include <fftw3.h>
#include "ps_lib.h"

void allocate_stats(statsStruct *stats, const paramsStruct params, int nz);
void free_stats(statsStruct stats, const paramsStruct params, int nz);
void write_statistics(const statsStruct stats, const paramsStruct params, int nz);
void shift(float *out, const float *in, int ofx, int ofy, int nx, int ny, int nz);

/* computation of moment or statistics */
float mean(const float *in, int N);
float compute_moment(const float *in, float m, int order, int N);
void min_and_max(float *m, float *M, const float *tab, int N);
float compute_skewness(const float *data_in, float m, float var, int N);
float compute_kurtosis(const float *data_in, float m, float var, int N);
void compute_auto_cor(float *Ac, const float *in, fftwf_complex *in_plan,
                      fftwf_complex *out_plan, fftwf_plan plan, fftwf_plan iplan,
                      int nx, int ny, int nz, int Na);
void compute_cross_cor(float *cross_cor, float **data, int N_data, int N, int nz);
void compute_cross_scale_cor(float *cross_scale_cor, float **data1, float **data2,
                             int N_data1, int N_data2, int N, int nz);

/* adjustment of the statistics */
void adjust_range(float *data, float m, float M, int N);
void adjust_mean_variance(float *data, float mean_out, float var_out, int N);
void adjust_skewness(float *data, float sk_out, int N);
void adjust_kurtosis(float *data, float ku_out, int N);
void adjust_auto_cor(float *data, const float *Ac, const float *var0,
                  const float *vari, fftwf_complex *in_plan,
                  fftwf_complex *out_plan, fftwf_plan plan, fftwf_plan iplan,
                  int nx, int ny, int nz, int Na, int scale);
void adjust_cross_cor(float **data, const float *cross_cor, int N_data, int N, int nz);
void adjust_covariance_color(float *data, const float cov[3][3], int N);
void adjust_cross_scale_cor(float **data1, float **data2, const float *cross_cor,
                         const float *cross_scale_cor, int N_data1, int N_data2,
                         int N, int nz);
void adjust_cross_scale_cor2(float *data1, float **data2,
                          float var, const float *cross_scale_cor,
                          int N_data, int N);

#endif
