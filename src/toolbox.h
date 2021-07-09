// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2021, Thibaud Briand <briand.thibaud@gmail.com>
// Copyright (C) 2021, Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
// All rights reserved.

/* toolbox.c contains several utility functions (Fourier related) */

#ifndef TOOLBOX_H
#define TOOLBOX_H

#include <fftw3.h>

void pointwise_complexfloat_multiplication(fftwf_complex *comp_out,
                                           const fftwf_complex *comp_in,
                                           const float *float_in,
                                           int N, int pd);
void precompute_plan(fftwf_plan *plan, fftwf_plan *iplan,
                     fftwf_complex *in_plan, fftwf_complex *out_plan,
                     int nx, int ny, int N_pyr);
void do_fft_plan(fftwf_plan plan, fftwf_complex *out_plan,
                 fftwf_complex *in_plan, fftwf_complex *out,
                 const fftwf_complex *in, int N, int nz);
void do_fft_plan_real(fftwf_plan plan, fftwf_complex *out_plan,
                      fftwf_complex *in_plan, fftwf_complex *out,
                      const float *in, int N, int nz);
void do_ifft_plan(fftwf_plan plan, fftwf_complex *out_plan,
                  fftwf_complex *in_plan, fftwf_complex *out,
                  const fftwf_complex *in, int N, int nz);
void do_ifft_plan_real(fftwf_plan plan, fftwf_complex *out_plan,
                       fftwf_complex *in_plan, float *out,
                       const fftwf_complex *in, int N, int nz);
void do_fft(fftwf_complex *out, const fftwf_complex *in,
            int nx, int ny, int nz);
void do_fft_real(fftwf_complex *out, const float *in, int nx, int ny, int nz);
void do_ifft(fftwf_complex *out, const fftwf_complex *in,
             int nx, int ny, int nz);
void do_ifft_real(float *out, const fftwf_complex *in, int nx, int ny, int nz);
void upsampling(fftwf_complex *out, const fftwf_complex *in,
                int nx, int ny, int nz);
void downsampling(fftwf_complex *out, const fftwf_complex *in,
                  int nx, int ny, int nz);

#endif
