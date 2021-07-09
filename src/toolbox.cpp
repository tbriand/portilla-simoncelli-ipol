// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2021, Thibaud Briand <briand.thibaud@gmail.com>
// Copyright (C) 2021, Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
// All rights reserved.

#include <stdlib.h>
#include <fftw3.h>
#include <string.h>

// Pointwise multiplication of a complex array by a float array
void pointwise_complexfloat_multiplication(fftwf_complex *comp_out,
                                           const fftwf_complex *comp_in,
                                           const float *float_in,
                                           int N, int pd)
{
  // loop over the channels
  for (int l = 0; l < pd; l++) {
    // multiplication
    for (int i = 0; i < N; i++) {
      comp_out[i + l*N][0] = comp_in[i + l*N][0] * float_in[i];
      comp_out[i + l*N][1] = comp_in[i + l*N][1] * float_in[i];
    }
  }
}

/* Interface with plan */

// Computation of the plans used for the FFT and iFFT
void precompute_plan(fftwf_plan *plan, fftwf_plan *iplan,
                     fftwf_complex *in_plan, fftwf_complex *out_plan,
                     int nx, int ny, int N_pyr) {
  // loop over the scales
  for (int i = 0; i <= N_pyr; i++) {
    // direct plan
    plan[i] = fftwf_plan_dft_2d (ny, nx, in_plan, out_plan,
                                 FFTW_FORWARD, FFTW_ESTIMATE);
    // inverse plan
    iplan[i] = fftwf_plan_dft_2d (ny, nx, in_plan, out_plan,
                                  FFTW_BACKWARD, FFTW_ESTIMATE);
    // update sizes
    nx /= 2;
    ny /= 2;
  }
}

// Compute the FFT of a complex array using a precomputed plan
void do_fft_plan(fftwf_plan plan, fftwf_complex *out_plan,
                 fftwf_complex *in_plan, fftwf_complex *out,
                 const fftwf_complex *in, int N, int nz)
{
  // loop over the channels
  for (int l = 0; l<nz; l++) {
    // copy to the input
    memcpy(in_plan, in + l*N, N*sizeof(fftwf_complex));

    // compute FFT
    fftwf_execute(plan);

    // copy to the output
    memcpy(out + l*N, out_plan, N*sizeof(fftwf_complex));
  }
}

// Compute the FFT of a real array using a precomputed plan
void do_fft_plan_real(fftwf_plan plan, fftwf_complex *out_plan,
                      fftwf_complex *in_plan, fftwf_complex *out,
                      const float *in, int N, int nz)
{
  // loop over the channels
  for (int l = 0; l < nz; l++) {
    // Real --> complex
    for(int i = 0; i < N; i++) {
      in_plan[i][0] = in[i + l*N];
      in_plan[i][1] = 0.0;
    }

    // compute FFT
    fftwf_execute(plan);

    // copy to the output
    memcpy(out + l*N, out_plan, N*sizeof(fftwf_complex));
  }
}

// Compute the iFFT of a complex array using a precomputed plan
void do_ifft_plan(fftwf_plan plan, fftwf_complex *out_plan,
                  fftwf_complex *in_plan, fftwf_complex *out,
                  const fftwf_complex *in, int N, int nz)
{
  // normalization constant
  float norm = 1.0/N;

  // loop over the channels
  for (int l = 0; l < nz; l++) {
    // copy to the input
    memcpy(in_plan, in + l*N, N*sizeof(fftwf_complex));

    // compute FFT
    fftwf_execute(plan);

    // normalization
    for(int i = 0; i < N; i++) {
      out[i + l*N][0] = out_plan[i][0]*norm;
      out[i + l*N][1] = out_plan[i][1]*norm;
    }
  }
}

// Compute the real part of the iFFT of a complex array using a precomputed plan
void do_ifft_plan_real(fftwf_plan plan, fftwf_complex *out_plan,
                       fftwf_complex *in_plan, float *out,
                       const fftwf_complex *in, int N, int nz)
{
  // normalization constant
  float norm = 1.0/N;

  // loop over the channels
  for (int l = 0; l < nz; l++) {
    // copy to the input
    memcpy(in_plan, in + l*N, N*sizeof(fftwf_complex));

    // compute fft
    fftwf_execute(plan);

    // complex --> real + normalization
    for(int i = 0; i < N; i++)
      out[i + l*N] = out_plan[i][0]*norm;
  }
}

/* Simpler interface (with the plan declaration inside) */

// Compute the FFT of a complex array
void do_fft(fftwf_complex *out, const fftwf_complex *in, int nx, int ny, int nz)
{
  // memory allocation
  fftwf_complex *in_plan = (fftwf_complex *)
    malloc(nx*ny*sizeof(fftwf_complex));
  fftwf_complex *out_plan = (fftwf_complex *)
    malloc(nx*ny*sizeof(fftwf_complex));
  fftwf_plan plan = fftwf_plan_dft_2d(ny, nx, in_plan, out_plan,
                                      FFTW_FORWARD, FFTW_ESTIMATE);

  // perform FFT from data and plan
  do_fft_plan(plan, out_plan, in_plan, out, in, nx*ny, nz);

  // free
  fftwf_destroy_plan(plan);
  fftwf_free(in_plan);
  fftwf_free(out_plan);
}

// Compute the fft of a real array
void do_fft_real(fftwf_complex *out, const float *in, int nx, int ny, int nz)
{
  // memory allocation
  fftwf_complex *in_plan = (fftwf_complex *)
    malloc(nx*ny*sizeof(fftwf_complex));
  fftwf_complex *out_plan = (fftwf_complex *)
    malloc(nx*ny*sizeof(fftwf_complex));
  fftwf_plan plan = fftwf_plan_dft_2d(ny, nx, in_plan, out_plan,
                                      FFTW_FORWARD, FFTW_ESTIMATE);

  // perform FFT from data and plan
  do_fft_plan_real(plan, out_plan, in_plan, out, in, nx*ny, nz);

  // free
  fftwf_destroy_plan(plan);
  fftwf_free(in_plan);
  fftwf_free(out_plan);
}

// Compute the iFFT of a complex array
void do_ifft(fftwf_complex *out, const fftwf_complex *in, int nx, int ny, int nz)
{
  // memory allocation
  fftwf_complex *in_plan = (fftwf_complex *)
    malloc(nx*ny*sizeof(fftwf_complex));
  fftwf_complex *out_plan = (fftwf_complex *)
    malloc(nx*ny*sizeof(fftwf_complex));
  fftwf_plan plan = fftwf_plan_dft_2d (ny, nx, in_plan, out_plan,
                                       FFTW_BACKWARD, FFTW_ESTIMATE);

  // perform iFFT from data and plan
  do_ifft_plan(plan, out_plan, in_plan, out, in, nx*ny, nz);

  // free
  fftwf_destroy_plan(plan);
  fftwf_free(in_plan);
  fftwf_free(out_plan);
}

// Compute the real part of the iFFT of a complex array
void do_ifft_real(float *out, const fftwf_complex *in, int nx, int ny, int nz)
{
  // memory allocation
  fftwf_complex *in_plan = (fftwf_complex *)
    malloc(nx*ny*sizeof(fftwf_complex));
  fftwf_complex *out_plan = (fftwf_complex *)
    malloc(nx*ny*sizeof(fftwf_complex));
  fftwf_plan plan = fftwf_plan_dft_2d (ny, nx, in_plan, out_plan,
                                       FFTW_BACKWARD, FFTW_ESTIMATE);

  // perform iFFT from data and plan
  do_ifft_plan_real(plan, out_plan, in_plan, out, in, nx*ny, nz);

  // free
  fftwf_destroy_plan(plan);
  fftwf_free(in_plan);
  fftwf_free(out_plan);
}

/* Resampling functions. See the following article for more details:
 * Thibaud Briand, Trigonometric Polynomial Interpolation of Images,
 * Image Processing On Line, 9 (2019), pp. 291–316.
 * http://doi.org/10.5201/ipol.2019.273
 */

// Upsampling by a factor 2 in the Fourier domain
// nx and ny are the small sizes i.e. BEFORE the zoom
void upsampling(fftwf_complex *out, const fftwf_complex *in,
                int nx, int ny, int nz)
{
  int r, l, rr, ll, k;

  int nx2 = (nx+1)/2; //index for the fftshift
  int ny2 = (ny+1)/2; //index for the fftshift

  // fill the output DFT with zeros
  for (l = 0; l < 4*nx*ny*nz; l++)
    out[l][0] = out[l][1] = 0.0;

  // fill the corners with the values
  for(r = 0; r < ny; r++) {
    rr = (r < ny2) ? r : r + ny;
    for(l = 0; l < nx; l++) {
      ll = (l < nx2) ? l : l + nx;
      for (k = 0; k < nz; k++) {
        out[ll + rr*2*nx + k*4*nx*ny][0] = 4*in[l + r*nx + k*nx*ny][0];
        out[ll + rr*2*nx + k*4*nx*ny][1] = 4*in[l + r*nx + k*nx*ny][1];
      }
    }
  }

  // correct the boundary values when the input image has even sizes
  if ( nx%2 == 0 ) {
    l = nx2; // positive in output and negative in input
    ll = nx2 + nx; // negative in output
    for(r = 0; r < ny; r++) {
      rr = (r < ny2) ? r : r + ny;
      for (k = 0; k < nz; k++) {
        out[ll + rr*2*nx + k*4*nx*ny][0] *= 0.5;
        out[ll + rr*2*nx + k*4*nx*ny][1] *= 0.5;
        out[l + rr*2*nx + k*4*nx*ny][0] = out[ll + rr*2*nx + k*4*nx*ny][0];
        out[l + rr*2*nx + k*4*nx*ny][1] = out[ll + rr*2*nx + k*4*nx*ny][1];
      }
    }
  }
  if ( ny%2 == 0 ) {
    r = ny2; // positive in output and negative in input
    rr = ny2 + ny; // negative in output
    for(l = 0; l < nx; l++) {
      ll = (l < nx2) ? l : l + nx;
      for (k = 0; k < nz; k++) {
        out[ll + rr*2*nx + k*4*nx*ny][0] *= 0.5;
        out[ll + rr*2*nx + k*4*nx*ny][1] *= 0.5;
        out[ll + r*2*nx + k*4*nx*ny][0] = out[ll + rr*2*nx + k*4*nx*ny][0];
        out[ll + r*2*nx + k*4*nx*ny][1] = out[ll + rr*2*nx + k*4*nx*ny][1];
      }
    }
  }

  if ( nx%2 == 0 && ny%2 == 0 ) {
    float tmp0, tmp1;
    l = nx2; // positive in output and negative in input
    ll = nx2 + nx; // negative in output
    r = ny2; // positive in output and negative in input
    rr = ny2 + ny; // negative in output
    for (k = 0; k < nz; k++) {
      tmp0 = in[l + r*nx + k*nx*ny][0]; // multiplied by 4/4
      tmp1 = in[l + r*nx + k*nx*ny][1]; // multiplied by 4/4 (should be 0)
      out[l + r*2*nx + k*4*nx*ny][0] = out[ll + r*2*nx + k*4*nx*ny][0]
        = out[l + rr*2*nx + k*4*nx*ny][0] = out[ll + rr*2*nx + k*4*nx*ny][0]
          = tmp0;
      out[l + r*2*nx + k*4*nx*ny][1] = out[ll + r*2*nx + k*4*nx*ny][1]
        = out[l + rr*2*nx + k*4*nx*ny][1] = out[ll + rr*2*nx + k*4*nx*ny][1]
          = tmp1;
    }
  }
}

// Downsampling by a factor 2 in the Fourier domain
// nx and ny are the small sizes i.e. AFTER the zoom
void downsampling(fftwf_complex *out, const fftwf_complex *in,
                  int nx, int ny, int nz)
{
  int r, l, rr, ll, k;

  int nx2 = (nx+1)/2; //index for the fftshift
  int ny2 = (ny+1)/2; //index for the fftshift
  float norm = 0.25;

  for(r = 0; r < ny; r++) {
    rr = (r < ny2) ? r : r + ny;
    for( l = 0; l < nx; l++){
      ll = (l < nx2) ? l : l + nx;
      for (k = 0; k < nz; k++) {
        out[l + r*nx + k*nx*ny][0] = in[ll + rr*2*nx + 4*k*nx*ny][0]*norm;
        out[l + r*nx + k*nx*ny][1] = in[ll + rr*2*nx + 4*k*nx*ny][1]*norm;
      }
    }
  }

  // correct the boundary values when the output image has even sizes
  if ( nx%2 == 0 ) {
    ll = nx2; // border index in output but opposite in input
    for (r = 0; r < ny; r++) {
      rr = (r < ny2) ? r : r + ny;
      for (k = 0; k < nz; k++) {
        out[ll + r*nx + k*nx*ny][0] += in[ll + rr*nx*2 + 4*k*nx*ny][0]*norm;
        out[ll + r*nx + k*nx*ny][1] += in[ll + rr*nx*2 + 4*k*nx*ny][1]*norm;
      }
    }
  }

  if ( ny%2 == 0 ) {
    rr = ny2;
    for (l = 0; l<nx; l++) {
      ll = (l<nx2) ? l : l + nx;
      for (k = 0; k < nz; k++) {
        out[l + rr*nx + k*nx*ny][0] += in[ll + rr*nx*2 + 4*k*nx*ny][0]*norm;
        out[l + rr*nx + k*nx*ny][1] += in[ll + rr*nx*2 + 4*k*nx*ny][1]*norm;
      }
    }
  }

  if ( nx%2 == 0 && ny%2 == 0 ) {
    l = nx2;
    ll = nx2 + nx;
    r = ny2;
    rr = ny2 + ny;
    for (k = 0; k < nz; k++) {
      out[l + r*nx + k*nx*ny][0] = norm*(in[l + 2*r*nx + 4*k*nx*ny][0]
        + in[ll + 2*r*nx + 4*k*nx*ny][0] + in[l + 2*rr*nx + 4*k*nx*ny][0]
          + in[ll + 2*rr*nx + 4*k*nx*ny][0]);
      out[l + r*nx + k*nx*ny][1] = norm*(in[l + 2*r*nx + 4*k*nx*ny][1]
        + in[ll + 2*r*nx + 4*k*nx*ny][1] + in[l + 2*rr*nx + 4*k*nx*ny][1]
          + in[ll + 2*rr*nx + 4*k*nx*ny][1]);
    }
  }
}
