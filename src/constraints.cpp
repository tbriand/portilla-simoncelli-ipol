/*
 * Copyright (C) 2021, Thibaud Briand, ENS Cachan <thibaud.briand@ens-cachan.fr>
 * Copyright (C) 2021, Jonathan Vacher, ENS Cachan <jvacher@ens-cachan.fr>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "ps_lib.h"
#include "toolbox.h"
#include "Eigen/Dense"
#include <unsupported/Eigen/Polynomials>
#include <Eigen/Eigenvalues>

using namespace Eigen;
using namespace std;

// Allocate memory for the sample statistics
void allocate_stats(statsStruct *stats, const paramsStruct params, int nz)
{
  int i;
  int N_pyr = params.N_pyr;
  int N_steer = params.N_steer;
  int Na = params.Na;

  stats->pixelStats = (float *) malloc(N_PIXELSTATS*nz*sizeof(float));
  stats->skewLow = (float *) malloc((1+N_pyr)*nz*sizeof(float));
  stats->kurtLow = (float *) malloc((1+N_pyr)*nz*sizeof(float));
  stats->varHigh = (float *) malloc(nz*sizeof(float));
  stats->magMeans = (float *) malloc(N_pyr*N_steer*nz*sizeof(float));
  stats->autoCorLow = (float **) malloc((1+N_pyr)*sizeof(float *));
  for(i = 0; i < 1+N_pyr; i++)
    stats->autoCorLow[i] = (float *) malloc(Na*Na*nz*sizeof(float));
  stats->autoCorMag = (float **) malloc(N_pyr*N_steer*sizeof(float *));
  for(i = 0; i < N_pyr*N_steer; i++)
    stats->autoCorMag[i] = (float *) malloc(Na*Na*nz*sizeof(float));
  stats->cousinMagCor = (float **) malloc(N_pyr*sizeof(float *));
  for(i = 0; i < N_pyr; i++)
    stats->cousinMagCor[i] = (float *)
      malloc(N_steer*N_steer*nz*nz*sizeof(float));
  stats->parentMagCor = (float **) malloc((N_pyr-1)*sizeof(float *));
  for(i = 0; i < N_pyr-1; i++)
    stats->parentMagCor[i] = (float *)
      malloc(N_steer*N_steer*nz*nz*sizeof(float));
  stats->parentRealCor = (float **) malloc(N_pyr*sizeof(float *));
  for(i = 0; i < N_pyr-1; i++)
    stats->parentRealCor[i] = (float *)
      malloc(2*N_steer*N_steer*nz*nz*sizeof(float));
  if (nz == 3) {
    stats->parentRealCor[N_pyr-1] = (float *)
      malloc(N_SMALLEST*N_steer*nz*nz*sizeof(float));
    stats->pixelStatsPCA = (float *) malloc(N_PIXELSTATSPCA*nz*sizeof(float));
    stats->autoCorPCA = (float *) malloc(Na*Na*nz*sizeof(float));
    stats->cousinRealCor = (float **) malloc((1+N_pyr)*sizeof(float *));
    for(i = 0; i < N_pyr; i++)
      stats->cousinRealCor[i] = (float *)
        malloc(N_steer*N_steer*nz*nz*sizeof(float));
    stats->cousinRealCor[N_pyr] = (float *)
      malloc(N_SMALLEST*N_SMALLEST*nz*nz*sizeof(float));
  }
}

// Free memory for the sample statistics
void free_stats(statsStruct stats, const paramsStruct params, int nz)
{
  int i;
  int N_pyr = params.N_pyr;
  int N_steer = params.N_steer;

  free(stats.pixelStats);
  free(stats.skewLow);
  free(stats.kurtLow);
  free(stats.varHigh);
  free(stats.magMeans);
  for(i = 0; i < 1+N_pyr; i++) {
    free(stats.autoCorLow[i]);
    if (nz == 3)
      free(stats.cousinRealCor[i]);
  }
  free(stats.autoCorLow);
  for(i = 0; i < N_pyr*N_steer; i++)
    free(stats.autoCorMag[i]);
  free(stats.autoCorMag);
  for(i = 0; i < N_pyr - 1; i++) {
    free(stats.parentMagCor[i]);
    free(stats.parentRealCor[i]);
  }
  if ( nz == 3)
    free(stats.parentRealCor[N_pyr-1]);
  for(i = 0; i < N_pyr; i++) {
    free(stats.cousinMagCor[i]);
  }
  free(stats.cousinMagCor);
  free(stats.parentMagCor);
  free(stats.parentRealCor);
  if( nz == 3 ) {
    free(stats.cousinRealCor);
    free(stats.pixelStatsPCA);
    free(stats.autoCorPCA);
  }
}

// Function for writing the statistics in a file
void write_statistics(const statsStruct stats, const paramsStruct params, int nz)
{
  int i, j, l, ind;
  int N_pyr = params.N_pyr;
  int N_steer = params.N_steer;
  int Na = params.Na;

  // open file
  FILE *fp = fopen("statistics_evolution.txt", "a");

  // summary statistics (i)(a)
  for (i = 0; i < 1+N_pyr; i++) {
    for (l = 0; l < nz; l++)
      fprintf(fp, " %f %f", stats.skewLow[i + (1+N_pyr)*l],
              stats.kurtLow[i + (1+N_pyr)*l]);
    fprintf(fp, ",");
  }

  // summary statistics (i)(b)
  for(l = 0; l < nz; l++)
    fprintf(fp, " %f", stats.varHigh[l]);
  fprintf(fp, ",");

  // summary statistics (i)(c)
  for (l = 0; l < nz; l++)
    for (i = 0; i < 6; i++)
      fprintf(fp, " %f", stats.pixelStats[i + N_PIXELSTATS*l]);
  fprintf(fp, ",");

  // summary statistics (ii)
  for (i = 0; i < 1+N_pyr; i++) {
    for(l = 0; l < Na*Na*nz; l++)
      fprintf(fp, " %f", stats.autoCorLow[i][l]);
    fprintf(fp, ",");
  }

  // summary statistics (iii)
  for(i = 0; i < N_pyr; i++) {
    for(j = 0; j < N_steer; j++) {
      ind = j + i*N_steer;
      for(l = 0; l < Na*Na*nz; l++)
        fprintf(fp, " %f", stats.autoCorMag[ind][l]);
      fprintf(fp, ",");
    }
  }

  // summary statistics (iv)
  for(i = 0; i < N_pyr; i++) {
    for(l = 0; l < N_steer*N_steer*nz*nz; l++)
      fprintf(fp, " %f", stats.cousinMagCor[i][l]);
    fprintf(fp, ",");
  }

  // summary statistics (v)
  for(i = 0; i < N_pyr - 1; i++) {
    for(l = 0; l < N_steer*N_steer*nz*nz; l++)
      fprintf(fp, " %f", stats.parentMagCor[i][l]);
    fprintf(fp, ",");
  }

  // summary statistics (vi)
  for(i = 0; i < N_pyr - 1; i++) {
    for(l = 0; l < 2*N_steer*N_steer*nz*nz; l++)
      fprintf(fp, " %f", stats.parentRealCor[i][l]);
    fprintf(fp, ",");
  }

  if( nz == 3 ) {
    // summary statistics (vii)
    for(l = 0; l < 3; l++)
      for(i = 0; i < 3; i++)
        fprintf(fp, " %f", stats.covariancePCA[l][i]);
    fprintf(fp, ",");

    // summary statistics (viii)
    for(i = 0; i < Na*Na*nz; i++)
      fprintf(fp, " %f", stats.autoCorPCA[i]);
    fprintf(fp, ",");

    // summary statistics (ix)
    for (l = 0; l < 3; l++)
      for (i = 0; i < 2; i++)
        fprintf(fp, " %f", stats.pixelStatsPCA[i + N_PIXELSTATSPCA*l]);
    fprintf(fp, ",");

    // summary statistics (x)
    for(i = 0; i < N_pyr; i++) {
      for(l = 0; l < N_steer*N_steer*nz*nz; l++)
        fprintf(fp, " %f", stats.cousinRealCor[i][l]);
      fprintf(fp, ",");
    }

    // summary statistics (xi)
    for(l = 0; l < N_SMALLEST*N_SMALLEST*nz*nz; l++)
      fprintf(fp, " %f", stats.cousinRealCor[N_pyr][l]);
    fprintf(fp, ",");

    // summary statistics (xii)
    for(l = 0; l < N_steer*N_SMALLEST*nz*nz; l++)
      fprintf(fp, " %f", stats.parentRealCor[N_pyr-1][l]);
    fprintf(fp, ",");
  }

  // close file
  fprintf(fp, "\n");
  fclose(fp);
}

// Apply an integer shift of (ofx, ofy) to an image
// out(i,j) = in(i - ofx, j - ofy)
// It is done using the nearest neighbor interpolation
// with periodic boundary condition
void shift(float *out, const float *in, int ofx, int ofy,
           int nx, int ny, int nz)
{
  int i, j, l, i2, j2;
  for(j = 0; j < ny; j++) {
    j2 = (j - ofy) % ny;
    j2 = (j2 < 0) ? j2 + ny : j2;
    for(i = 0; i < nx; i++) {
      i2 = (i - ofx) % nx;
      i2 = (i2 < 0) ? i2 + nx : i2;
      for(l = 0; l < nz; l++)
          out[i+j*nx+l*nx*ny] = in[i2 + j2*nx + l*nx*ny];
    }
  }
}

/* moment part */

// Compute the mean of an array
float mean(const float *data, int N)
{
  double m = 0.0;

  // sum loop
  for(int i = 0; i < N; i++)
    m += data[i];

  // normalization
  m /= N;

  return m;
}

// Compute the moment of a given order of an array (Equation 39)
// To save computations the mean is computed outside this function
float compute_moment(const float *data, float m, int order, int N)
{
  double moment = 0.0;
  double tmp, tmp2;

  // sum loop
  for(int i = 0; i < N; i++) {
    tmp = 1.0;
    tmp2 = data[i] - m;
    for(int j = 0; j < order; j++)
      tmp *= tmp2;
    moment += tmp;
  }

  // normalization
  moment /= N;

  return moment;
}

// Compute the skewness of an array (Equation 40)
// Set the value to 0 for a constant image
float compute_skewness(const float *data_in, float m, float var, int N)
{
  float order_3 = compute_moment(data_in, m, 3, N);
  float skewness = (var > 0) ? order_3/sqrt(var*var*var) : 0;

  return skewness;
}

// Compute the kurtosis of an array (Equation 40)
// Set the value to 3 for a constant image
float compute_kurtosis(const float *data_in, float m, float var, int N)
{
  float order_4 = compute_moment(data_in, m, 4, N);
  float kurtosis = (var > 0) ? order_4/(var*var) : 3;

  return kurtosis;
}

/* range part */

// Compute the min and max values of an array
void min_and_max(float *m, float *M, const float *tab, int N)
{
  *m = *M = tab[0];

  for(int i = 1; i < N; i++) {
    if (tab[i] < *m) *m = tab[i];
    if (tab[i] > *M) *M = tab[i];
  }
}

// Adjust the range of an array so that it fits in [m;M] (Appendix A.1.1)
// This corresponds to Equation 49
void adjust_range(float *data, float m, float M, int N)
{
  // loop on the array
  for(int i = 0; i < N; i++) {
    // projection in [m,M]
    if (data[i] < m)
      data[i] = m;
    else if (data[i] > M)
      data[i] = M;
  }
}

/* Modification part of marginal statistics */

// Adjust the mean and the variance of an array (Appendix A.1.2)
// This corresponds to Equation 50
void adjust_mean_variance(float *data, float mean_out, float var_out, int N)
{
  // compute the input mean and variance
  float m = mean(data, N);
  float var_in = compute_moment(data, m, 2, N);

  float factor = (var_in > 0) ? sqrt(var_out/var_in) : 1;
  // loop on the array
  for(int i = 0; i < N; i++)
    data[i] = factor*(data[i]-m) + mean_out;
}

// Adjust the skewness of an array (Appendix A.1.3)
// This corresponds to Algorithm 6
// The input data is assumed to have a zero mean
void adjust_skewness(float *data, float sk_out, int N)
{
  int i;

  // computation of the moments (Line 1)
  // this is done outside of the function compute_moment to save computations
  double m2 = 0.0;
  double m3 = 0.0;
  double m4 = 0.0;
  double m5 = 0.0;
  double m6 = 0.0;
  double tmp, tmp2;
  for (i = 0; i < N; i++) {
    tmp2 = tmp = data[i];

    tmp2 *= tmp;
    m2 += tmp2;

    tmp2 *= tmp;
    m3 += tmp2;

    tmp2 *= tmp;
    m4 += tmp2;

    tmp2 *= tmp;
    m5 += tmp2;

    tmp2 *= tmp;
    m6 += tmp2;
  }
  tmp = 1.0/N;
  m2 *= tmp;
  m3 *= tmp;
  m4 *= tmp;
  m5 *= tmp;
  m6 *= tmp;

  // compute skewness of the input (Line 2)
  double std = sqrt(m2);
  double sk_in = m3/(std*std*std);

  // security check
  double snr = 20*log10(fabs(sk_out/(sk_out-sk_in)));

  // do nothing if the skewness are too close
  if ( snr <= 60 ) {
    // polynomial coefficients computation (Line 3)
    // coefficients of the numerator P (Equation 56)
    double p0 = sk_in*m2*std;
    double p1 = 3 * ( m4 - m2 * m2 * ( 1 + sk_in*sk_in ) );
    double p2 = 3 * ( m5 - 2*std*sk_in*m4 + m2*m2*std*sk_in*sk_in*sk_in );
    double p3 = m6 - 3*std*sk_in*m5 + 3*m2*(sk_in*sk_in - 1)*m4
                + m2*m2*m2*(2 + 3*sk_in*sk_in - sk_in*sk_in*sk_in*sk_in);
    Vector4d poly_num;
    poly_num << p0, p1, p2, p3;

    // coefficients of the denominator Q (Equation 56)
    double q0 = m2;
    double q1 = 0;
    double q2 = m4 - (1 + sk_in*sk_in)*m2*m2;
    Vector3d poly_denom;
    poly_denom << q0, q1, q2;

    // coefficients of Q^3(X^2) (Equation 60)
    double b0 = q0*q0*q0;
    double b2 = 3*q0*q0*q2;
    double b4 = 3*q0*q2*q2;
    double b6 = q2*q2*q2;
    VectorXd b_poly(4);
    b_poly << b0, b2, b4, b6;

    // derivative with respect to lambda (Equation 62)
    double d0 = p1*b0;
    double d1 = -p0*b2 + 2*p2*b0;
    double d2 = 3*p3*b0;
    double d3 = -2*p0*b4 + p2*b2;
    double d4 = -p1*b4 + 2*p3*b2;
    double d5 = -3*p0*b6;
    double d6 = -2*p1*b6 + p3*b4;
    double d7 = -p2*b6;
    VectorXd derivative(8);
    derivative << d0, d1, d2, d3, d4, d5, d6, d7;

    // compute the roots of d (Line 4)
    PolynomialSolver< double, 7 > dsolve( derivative );

    // find the minimal and maximal skewness reachable (Line 5 and Line 6)
    double lneg = - 1e6;
    double lpos = 1e6;
    double real_part;
    for (i = 0; i < 7; i++) {
      real_part = dsolve.roots()[i].real();
      if ( fabs(dsolve.roots()[i].imag()/real_part)<1e-6 ) {
        if ( real_part < 0 && real_part > lneg )
          lneg = real_part;
        else if ( real_part > 0 && real_part < lpos )
          lpos = real_part;
      }
    }
    double skmin = poly_eval(poly_num, lneg)/sqrt(poly_eval(b_poly, lneg*lneg));
    double skmax = poly_eval(poly_num, lpos)/sqrt(poly_eval(b_poly, lpos*lpos));

    // solve for lambda (Line 7 to 19)
    double lambda = 0.0;
    if ( sk_out <= skmin ) // saturating down the skewness (Line 8 and Line 9)
      lambda = lneg;
    else if ( sk_out >= skmax ) // saturating up the skewness (Line 10 and Line 11)
      lambda = lpos;
    else {
      // define the polynomial A = P^2 - sk_out^2*Q^2 (Equation 58)
      // This is done in Line 3
      double sk_out2 = sk_out * sk_out;
      double a0 = p0*p0 - sk_out2*b0;
      double a1 = 2*p1*p0;
      double a2 = p1*p1 + 2*p2*p0 - sk_out2*b2;
      double a3 = 2*(p3*p0 + p1*p2);
      double a4 = p2*p2 + 2*p3*p1 - sk_out2*b4;
      double a5 = 2*p3*p2;
      double a6 = p3*p3 - sk_out2*b6;

      // declare the polynomials
      VectorXd poly(7);
      poly << a0, a1, a2, a3, a4, a5, a6;

      // solve the polynomial (Line 13)
      PolynomialSolver< double, 6 > psolve( poly );

      // keep the real roots (Line 14)
      double real_roots[6] = {0};
      int p = 0; // number of real roots
      for(i = 0; i < 6; i++) {
        real_part = psolve.roots()[i].real();
        if ( fabs(psolve.roots()[i].imag()/real_part) < 1e-6 )
          real_roots[p++]= real_part;
      }

      if ( p == 1 ) // if only one solution left (Line 15 and Line 16)
        lambda = real_roots[0];
      else if ( p > 1 ) { // if several solutions left (Line 17)
        // keep the roots giving the numerator of good sign (Line 18)
        // if the sign is 0 it is accepted
        int q = 0; // number of acceptable solution
        double final_roots[6]={0};
        int sign0 = ( fabs(sk_out)<1e-6 ) ? 0 : (sk_out>0)-(sk_out<0);
        int sign;
        double numerator;
        for(i = 0; i < p; i++) {
          numerator = poly_eval(poly_num, real_roots[i]);
          sign = ( fabs(numerator)<1e-6 ) ? 0 : (numerator > 0)-(numerator < 0);
          if( sign == sign0 || sign*sign0 == 0 )
            final_roots[q++]= real_roots[i];
        }

        // get the solution with minimal modulus (Line 19)
        if(q > 0) {
          lambda = final_roots[0];
          double modulus = fabs(lambda);
          for (i = 1; i < q; i++) {
            double modulus2 = fabs(final_roots[i]);
            if ( modulus2 < modulus) {
              lambda = final_roots[i];
              modulus = modulus2;
            }
          }
        }
      }
    }

    // gradient descent (Line 20)
    double g;
    double stdsk = std*sk_in;
    for(i = 0; i < N ; i++) {
      g = data[i]*(data[i]-stdsk)-m2;
      data[i] += lambda*g;
    }

    // adjust the mean and the variance (to insure the stability of the code)
    adjust_mean_variance(data, 0.0, m2, N);
  }
}

// Adjust the kurtosis of an array (Appendix A.1.4)
// This corresponds to Algorithm 7
// The input data is assumed to have a zero mean
void adjust_kurtosis(float *data, float ku_out, int N)
{
  int i;

  // computation of the moments (Line 1)
  // this is done outside of compute_moment to save computations
  double m2 = 0.0;
  double m3 = 0.0;
  double m4 = 0.0;
  double m5 = 0.0;
  double m6 = 0.0;
  double m7 = 0.0;
  double m8 = 0.0;
  double m9 = 0.0;
  double m10 = 0.0;
  double m12 = 0.0;
  double tmp, tmp2;
  for (i = 0; i < N; i++) {
    tmp2 = tmp = data[i];

    tmp2 *= tmp;
    m2 += tmp2;

    tmp2 *= tmp;
    m3 += tmp2;

    tmp2 *= tmp;
    m4 += tmp2;

    tmp2 *= tmp;
    m5 += tmp2;

    tmp2 *= tmp;
    m6 += tmp2;

    tmp2 *= tmp;
    m7 += tmp2;

    tmp2 *= tmp;
    m8 += tmp2;

    tmp2 *= tmp;
    m9 += tmp2;

    tmp2 *= tmp;
    m10 += tmp2;

    tmp2 *= tmp*tmp;
    m12 += tmp2;
  }
  tmp = 1.0/N;
  m2 *= tmp;
  m3 *= tmp;
  m4 *= tmp;
  m5 *= tmp;
  m6 *= tmp;
  m7 *= tmp;
  m8 *= tmp;
  m9 *= tmp;
  m10 *= tmp;
  m12 *= tmp;

  // compute the kurtosis of the input (Line 2)
  double ku_in = m4/(m2*m2);

  // security check
  double snr = 20*log10(fabs(ku_out/(ku_out-ku_in)));
  if ( snr <= 60 ) { // do nothing if the kurtoses are too close
    // polynomial coefficients computation (Line 3)
    // auxilary useful variable
    double alpha = m4/m2;

    // define the coefficients of the numerator (Line 67)
    double p0 = m4;
    double p1 = 4 * ( m6 - alpha * alpha * m2 - m3 * m3 );
    double p2 = 6 * ( m8 - 2 * alpha * m6 - 2 * m3 * m5 + alpha * alpha * m4
      + (m2 + 2*alpha) * m3 * m3 );
    double p3 = 4 * ( m10 - 3 * alpha * m8 - 3 * m3 * m7
      + 3 * alpha * alpha * m6 + 6 * alpha * m3 * m5
      + 3 * m3 * m3 * m4 - alpha * alpha * alpha * m4
      - 3 * alpha * alpha * m3 * m3 - 3 * m4 * m3 * m3 );
    double p4 =  m12 - 4 * alpha * m10 - 4 * m3 * m9
      + 6 * alpha * alpha * m8 + 12 * alpha * m3 * m7 +  6 * m3 * m3 * m6
      - 4 * alpha * alpha * alpha * m6 - 12 * alpha * alpha * m3 * m5
      + alpha * alpha * alpha * alpha * m4 - 12 * alpha * m3 * m3 * m4
      + 4 * alpha * alpha * alpha * m3 * m3 + 6 * alpha * alpha * m3 * m3 * m2
      - 3 * m3 * m3 * m3 * m3;
    VectorXd poly_num(5);
    poly_num << p0, p1, p2, p3, p4;

    // define the coefficients of the denominator (Line 67)
    double q0 = m2;
    double q1 = 0;
    double q2 = p1*0.25;
    Vector3d poly_denom;
    poly_denom << q0, q1, q2;

    // derivative with respect to lambda (Line 71)
    double d0 = p1*q0;
    double d1 = -4*q2*p0 + 2*p2*q0;
    double d2 = -3*q2*p1 + 3*p3*q0;
    double d3 = -2*p2*q2 + 4*p4*q0;
    double d4 = -p3*q2;
    VectorXd derivative(5);
    derivative << d0, d1, d2, d3, d4;

    // compute the roots of d (Line 4)
    PolynomialSolver< double, 4 > dsolve( derivative );

    // find the minimal and maximal kurtosis reachable (Line 5 and Line 6)
    double lneg = - 1e6;
    double lpos = 1e6;
    double real_part;
    for (i = 0; i<4; i++) {
      real_part = dsolve.roots()[i].real();
      if ( fabs(dsolve.roots()[i].imag()/real_part) < 1e-6 ) {
        if ( real_part < 0 && real_part > lneg )
          lneg = real_part;
        else if ( real_part > 0 && real_part < lpos )
          lpos = real_part;
      }
    }
    tmp = poly_eval(poly_denom, lneg);
    double kumin = poly_eval(poly_num, lneg)/(tmp*tmp);
    tmp = poly_eval(poly_denom, lpos);
    double kumax = poly_eval(poly_num, lpos)/(tmp*tmp);

    // solves for lambda (Line 7 to 15)
    double lambda = 0.0;
    if ( ku_out <= kumin ) // saturating down the skewness (Line 8 and Line 9)
      lambda = lneg;
    else if ( ku_out >= kumax ) // saturating up the skewness (Line 10 and Line 11)
      lambda = lpos;
    else {
      // define the polynomial (Equation 69)
      // this is done in Line 3
      double a4 = p4 - ku_out*q2*q2;
      double a3 = p3;
      double a2 = p2 - 2*ku_out*q0*q2;
      double a1 = p1;
      double a0 = p0 - ku_out*q0*q0;
      VectorXd poly(5);
      poly << a0, a1, a2, a3, a4;

      // compute the roots of a (Line 13)
      PolynomialSolver< double, 4 > psolve( poly );

      // keep the real roots (Line 14)
      double real_roots[4] = {0};
      int p = 0; // number of real roots
      for(i = 0; i < 4; i++) {
        real_part = psolve.roots()[i].real();
        if ( fabs(psolve.roots()[i].imag()/real_part) < 1e-6 )
            real_roots[p++] = real_part;
      }

      // if acceptable solutions get the one with smallest modulus (Line 15)
      if ( p > 0 ) {
        lambda = real_roots[0];
        double modulus = fabs(lambda);
        for (i = 1; i < p; i++) {
          double modulus2 = fabs(real_roots[i]);
          if ( modulus2 < modulus) {
            lambda = real_roots[i];
            modulus = modulus2;
          }
        }
      }
    }

    // gradient descent (Line 16)
    double g;
    for(i = 0; i < N ; i++) {
      g = data[i] * (data[i] * data[i] - alpha ) - m3;
      data[i] += lambda * g;
    }

    // adjust the mean and the variance
    adjust_mean_variance(data, 0.0, m2, N);
  }
}

/* Correlation part */

// Compute the central part of the auto-correlation of an image (Equation 42)
// Na is assumed to be an odd number smaller than nx and ny
// The input data is assumed to have a zero mean
void compute_auto_cor(float *Ac, const float *in, fftwf_complex *in_plan,
                      fftwf_complex *out_plan, fftwf_plan plan,
                      fftwf_plan iplan, int nx, int ny, int nz, int Na)
{
  int i, j, l;

  // memory allocation
  fftwf_complex *fft = (fftwf_complex *)
    fftwf_malloc( nx*ny*nz * sizeof(fftwf_complex));
  float *full_auto_cor = (float *) malloc(nx*ny*nz*sizeof(float));

  // compute the fft
  do_fft_plan_real(plan, out_plan, in_plan, fft, in, nx*ny, nz);

  // squared modulus
  for(i = 0; i < nx*ny*nz; i++) {
    fft[i][0] = fft[i][0]*fft[i][0] + fft[i][1]*fft[i][1];
    fft[i][1] = 0.0;
  }

  // compute the inverse fft
  do_ifft_plan_real(iplan, out_plan, in_plan, full_auto_cor, fft, nx*ny, nz);

  // keep the central part of size Na*Na
  int hNa = (Na-1)/2;
  float ifactor = 1.0/(nx*ny);
  int ind;
  for(i = 0; i < Na ;i++) {
    for(j = 0; j < Na; j++) {
      if((i < hNa) && (j < hNa))
        ind = nx - hNa + i + (ny-hNa+j)*nx;
      else if((i < hNa) && (j> hNa-1))
        ind = nx - hNa + i + (j-hNa)*nx;
      else if((i > hNa-1) && (j < hNa))
        ind = i - hNa + (ny-hNa+j)*nx;
      else
        ind = i - hNa + (j-hNa)*nx;

      for(l = 0; l < nz; l++)
        Ac[i + j*Na + l*Na*Na] = full_auto_cor[ind + l*nx*ny]*ifactor;
    }
  }

  // free memory
  fftwf_free(fft);
  free(full_auto_cor);
}

// Compute the pairwise cross-correlation matrix (Equation 43)
// The input data is assumed to have a zero mean
void compute_cross_cor(float *cross_cor, float **data, int N_data,
                       int N, int nz)
{
  int i, j, l;
  MatrixXf M(N_data*nz, N);

  // put data in a matrix M
  for(i = 0; i < N_data; i++)
    for(j = 0; j < N; j++)
      for(l = 0; l < nz; l++)
        M(i + l*N_data,j) = data[i][j + l*N];

  // compute the cross-correlation
  MatrixXf tmp = M*(M.transpose())/N;

  // put the cross-correlation in the ouput array
  for(i = 0; i < N_data*nz; i++)
    for(j = 0; j < N_data*nz; j++)
      cross_cor[j + i*N_data*nz] = tmp(i,j);
}

// Compute the cross-correlation matrix (Equation 44)
// The input data is assumed to have a zero mean
void compute_cross_scale_cor(float *cross_scale_cor, float **data1,
                             float **data2, int N_data1, int N_data2,
                             int N, int nz)
{
  int i, j, l;
  MatrixXf X(N_data1*nz, N);
  MatrixXf Y(N_data2*nz, N);

  // put data in a matrix X
  for(i = 0; i < N_data1; i++)
    for(j = 0; j < N; j++)
      for(l = 0; l<nz; l++)
        X(i + l*N_data1,j)= data1[i][j + l*N];

  // put data in a matrix Y
  for(i = 0; i < N_data2; i++)
    for(j = 0; j < N; j++)
      for(l = 0; l<nz; l++)
        Y(i + l*N_data2,j) = data2[i][j + l*N];

  MatrixXf tmp = X*(Y.transpose())/N;

  // put the cross-correlation in the ouput array
  for(i = 0; i < N_data1*nz; i++)
    for(j = 0; j < N_data2*nz; j++)
      cross_scale_cor[j + i*N_data2*nz] = tmp(i,j);
}

// Adjust the central part of the auto-correlation (Appendix A.2)
// This corresponds to Algorithm 8
// The input data is assumed to have a zero mean
void adjust_auto_cor(float *data, const float *Ac, const float *var0,
                  const float *vari, fftwf_complex *in_plan,
                  fftwf_complex *out_plan, fftwf_plan plan, fftwf_plan iplan,
                  int nx, int ny, int nz, int Na, int scale)
{
  int i, j, k, l, p;

  // central location
  int hNa = ((Na-1)/2);

  // memory allocation
  fftwf_complex *fft_data = (fftwf_complex *)
    fftwf_malloc( nx*ny*nz * sizeof(fftwf_complex));
  fftwf_complex *fft_tmp = (fftwf_complex *)
    fftwf_malloc( nx*ny*nz * sizeof(fftwf_complex));
  float *tmp = (float *) malloc(nx*ny*nz*sizeof(float));
  int Na2 = 2*Na - 1;
  float *Ac_in = (float *) malloc(Na2*Na2*nz*sizeof(float));

  // compute the fft of the input (Line 1)
  do_fft_plan_real(plan, out_plan, in_plan, fft_data, data, nx*ny, nz);

  // computation of the auto-correlation image (Line 2)
  // square the modulus of the DFT
  for(i = 0; i<nx*ny*nz; i++) {
    fft_tmp[i][0] = fft_data[i][0]*fft_data[i][0]
      + fft_data[i][1]*fft_data[i][1];
    fft_tmp[i][1] = 0.0;
  }

  // compute inverse fft
  do_ifft_plan_real(iplan, out_plan, in_plan, tmp, fft_tmp, nx*ny, nz);

  // keep the central part of size (2*Na-1)*(2*Na-1) and renormalize
  int hNa2 = (Na2-1)/2;
  float ifactor = 1.0/(nx*ny);
  int ind;
  for(i = 0; i < Na2 ; i++) {
    for(j = 0; j < Na2; j++) {
      if((i < hNa2) && (j < hNa2))
        ind = nx - hNa2 + i + (ny-hNa2+j)*nx;
      else if((i < hNa2) && (j > hNa2-1))
        ind = nx - hNa2 + i + (j-hNa2)*nx;
      else if((i > hNa2-1) && (j < hNa2))
        ind = i - hNa2 + (ny-hNa2+j)*nx;
      else
        ind = i - hNa2 + (j-hNa2)*nx;

      for(l = 0; l < nz; l++)
        Ac_in[i + j*Na2 + l*Na2*Na2] = tmp[ind + l*nx*ny]*ifactor;
    }
  }

  // declaration for the following computations
  int t = ((Na*Na+1)/2);
  MatrixXf A(t,t);
  VectorXf B(t);
  MatrixXf M(Na,Na);
  MatrixXf rM(Na,Na);
  MatrixXf auto_cor_in(Na2,Na2);
  MatrixXf auto_cor_out(Na,Na);
  int end_loop;
  float factor;
  float tol = (nz == 3) ? 1e-3 : 1e-4;

  // loop over the channels
  for(l = 0; l < nz; l++) {
    if ( vari[l]/var0[l] > tol ) { // perform adjustment
      // put auto-correlation arrays in matrices
      for(i = 0; i < Na2; i++)
        for(j = 0; j < Na2; j++)
          auto_cor_in(j,i) = Ac_in[i + j*Na2 + l*Na2*Na2];
      for(i = 0; i < Na; i++)
        for(j = 0; j < Na; j++)
          auto_cor_out(j,i) = Ac[i + j*Na + l*Na*Na];

      // build the matrices involved in Equation 76
      // A <--> R(v)
      // B <--> R(u)
      for(k = hNa; k < Na; k++) {
        end_loop = (k < Na-1) ? hNa + Na : Na;
        for(p = hNa; p<end_loop; p++) {
          M = auto_cor_in.block(k - hNa, p - hNa, Na, Na);

          for(i = 0; i < Na; i++)
            for(j = 0; j < Na; j++)
              rM(i,j) = M(i,j) + M(Na-1-i,Na-1-j);
          rM(hNa, hNa) /= 2;
          rM.resize(Na*Na,1);
          for(i = 0; i < t; i++)
            A((k - hNa)*Na + p - hNa,i)=rM(i,0);
          rM.resize(Na,Na);
          B((k - hNa)*Na + p - hNa) = auto_cor_out(k - hNa,p - hNa);
        }
      }

      // Solve Equation 76 (Line 3)
      // A * sol = B
      VectorXf sol = A.colPivHouseholderQr().solve(B);

      // Rearrange indices to build the center of R(h_lambda)
      MatrixXf fullsol(2*t-1,1);
      for(i = 0; i < t; i++) {
        fullsol(i,0)=sol(i);
        if(i<t-1)
          fullsol(t + i,0) = sol(t - i -2);
      }
      fullsol.resize(Na,Na);

      // Pad the center with zeros
      MatrixXf hsquared0 = MatrixXf::Zero(ny,nx);
      hsquared0.block(ny/2-hNa,nx/2-hNa, Na, Na) = fullsol;
      MatrixXf shiftedhsquared0(ny,nx);
      shiftedhsquared0 << hsquared0.block(ny/2,nx/2,ny/2,nx/2),
        hsquared0.block(ny/2,0,ny/2,nx/2), hsquared0.block(0,nx/2,ny/2,nx/2),
        hsquared0.block(0,0,ny/2,nx/2);

      // store the matrix in an array before computing its DFT
      for(j = 0; j < ny; j++)
        for(i = 0; i < nx; i++)
          tmp[i + j*nx + l*nx*ny] = shiftedhsquared0(j,i);

      // compute the dft of the computed array (Line 4)
      do_fft_plan_real(plan, out_plan, in_plan, fft_tmp + l*nx*ny,
                       tmp + l*nx*ny, nx*ny, 1);

      // compute the convolution in Fourier domain (Line 5 and Equation 75)
      for(i = 0; i < nx*ny; i++) {
        // compute the square root of the absolute value of the real part
        factor = sqrt(fabs(fft_tmp[i + l*nx*ny][0]));

        // multiplication in Fourier domain
        fft_data[i + l*nx*ny][0] *= factor;
        fft_data[i + l*nx*ny][1] *= factor;
      }

      // compute the inverse fft (Line 6)
      do_ifft_plan_real(iplan, out_plan, in_plan, data + l*nx*ny,
                        fft_data + l*nx*ny, nx*ny, 1);
    }
    else { // variance adjustment
      adjust_mean_variance(data + l*nx*ny, 0.0, vari[l]/pow(16, scale), nx*ny);
    }
  }

  // free memory
  fftwf_free(fft_data);
  fftwf_free(fft_tmp);
  free(tmp);
  free(Ac_in);
}

// Adjust the pairwise cross-correlation of a list of sub-bands (Appendix A.3)
// This corresponds to Algorithm 9
// Linearly adjust variables in data to have the pairwise cross-correlation cross_cor
// The input data is assumed to have a zero mean
void adjust_cross_cor(float **data, const float *cross_cor,
                   int N_data, int N, int nz)
{
  int i, j, l;
  MatrixXf V(N_data*nz, N);

  // put data in a matrix V
  for(i = 0; i < N_data; i++)
    for(j = 0; j < N; j++)
      for(l = 0; l < nz; l++)
        V(i + l*N_data,j) = data[i][j + l*N];

  // store the target cross-correlation in a matrix
  MatrixXf tildeC(N_data*nz, N_data*nz);
  for(i = 0; i < N_data*nz; i++)
    for(j = 0; j < N_data*nz; j++)
      tildeC(i,j) = cross_cor[j + i*N_data*nz];

  // compute the input pairwise cross-correlation (Line 1)
  MatrixXf C = V*(V.transpose())/N;

  // eigen decomposition to compute the square root of matrices
  SelfAdjointEigenSolver<MatrixXf> eigensolver_in(C);
  SelfAdjointEigenSolver<MatrixXf> eigensolver_out(tildeC);
  MatrixXf D_in = eigensolver_in.eigenvalues().asDiagonal();
  MatrixXf P_in = eigensolver_in.eigenvectors();
  MatrixXf D_out = eigensolver_out.eigenvalues().asDiagonal();
  MatrixXf P_out = eigensolver_out.eigenvectors();

  // square root and inverse square root
  // isD_in = D_in^{-1/2}
  // sD_out = D_out^{1/2}
  MatrixXf isD_in = MatrixXf::Zero(N_data*nz,N_data*nz);
  MatrixXf sD_out = MatrixXf::Zero(N_data*nz,N_data*nz);
  int test1 = 0, test2 = 0; // to test if the matrices stay at 0 or not
  for(i = 0; i < N_data*nz; i++) {
    if( D_in(i,i) > 1e-12 ) {
      isD_in(i,i) = 1.0/sqrt(D_in(i,i));
      test1 = 1;
    }
    if( D_out(i,i) > 0) {
      sD_out(i,i) = sqrt(D_out(i,i));
      test2 = 1;
    }
  }

  if (test1 && test2) { // if both matrices have non-zero values
    MatrixXf Lambda = P_out*sD_out*(P_out.transpose())*P_in*isD_in*(P_in.transpose());

    // update the matrix V (Line 3)
    V = Lambda*V;

    // update data
    for(i = 0; i < N_data; i++) {
      for(j = 0; j < N; j++)
        for(l = 0; l < nz; l++)
          data[i][j + l*N] = V(i + l*N_data,j);
    }
  }
}

// Adjust the color covariance matrix of PCA bands (Appendix A.3)
// This corresponds to Algorithm 9
// Linearly adjust variables in data to have the covariance matrix cov
// The computations are similar as in adjust_cross_cor but the structure
// of data is different
void adjust_covariance_color(float *data, const float cov[3][3], int N)
{
  int i, j;
  MatrixXf U(3, N);

  // put data in a matrix U
  for(i = 0; i < 3; i++)
    for(j = 0; j < N; j++)
      U(i,j) = data[j + i*N];

  // store the target cross-correlation in a matrix
  Matrix3f tildeC;
  for(i = 0; i < 3; i++)
    for(j = 0; j < 3; j++)
      tildeC(i,j) = cov[i][j];

  // compute the covariance matrix (Line 1)
  Matrix3f C = U*(U.transpose())/N;

  // eigen decomposition to compute the square root of matrices
  SelfAdjointEigenSolver<Matrix3f> eigensolver_in(C);
  SelfAdjointEigenSolver<Matrix3f> eigensolver_out(tildeC);
  Matrix3f D_in = eigensolver_in.eigenvalues().asDiagonal();
  Matrix3f P_in = eigensolver_in.eigenvectors();
  Matrix3f D_out = eigensolver_out.eigenvalues().asDiagonal();
  Matrix3f P_out = eigensolver_out.eigenvectors();

  // square root and inverse square root
  // isD_in = D_in^{-1/2}
  // sD_out = D_out^{1/2}
  Matrix3f isD_in = MatrixXf::Zero(3,3);
  Matrix3f sD_out = MatrixXf::Zero(3,3);
  int test1 = 0, test2 = 0; // to test if the matrices stay at 0 or not
  for(i = 0; i < 3; i++) {
    if( D_in(i,i) > 1e-12 ) {
      isD_in(i,i) = 1.0/sqrt(D_in(i,i));
      test1 = 1;
    }
    if( D_out(i,i) > 0 ) {
      sD_out(i,i) = sqrt(D_out(i,i));
      test2 = 1;
    }
  }

  if (test1 && test2) { // if both matrices have non-zero values
    // compute the solution matrix (Line 2)
    MatrixXf Lambda = P_out*sD_out*(P_out.transpose())*P_in*isD_in*(P_in.transpose());

    // update the matrix U (Line 3)
    U = Lambda*U;

    // update data
    for(i = 0; i < 3; i++) {
      for(j = 0; j < N; j++)
        data[j + i*N] = U(i,j);
    }
  }
}

// Adjust the cross-correlations of a list of sub-bands given a list
// of fixed sub-bands (Appendix A.3)
// This corresponds to Algorithm 10
// Linearly adjust variables in data1 to have the pairwise cross-correlation
// cross_cor and the cross-correlation cross_scale_cor
// The input data is assumed to have a zero mean
// Note that data2 is not modified
void adjust_cross_scale_cor(float **data1, float **data2, const float *cross_cor,
                         const float *cross_scale_cor, int N_data1, int N_data2,
                         int N, int nz)
{
  int i, j, l;
  MatrixXf V(N_data1*nz, N);
  MatrixXf W(N_data2*nz, N);

  float tol = (nz == 3) ? 1e-3 : 1e-6;

  // put data in a matrix V
  for(i = 0; i < N_data1; i++)
    for(j = 0; j < N; j++)
      for(l = 0; l < nz; l++)
        V(i + l*N_data1,j) = data1[i][j + l*N];

  // put data in a matrix W
  for(i = 0; i < N_data2; i++)
    for(j = 0; j < N; j++)
      for(l = 0; l < nz; l++)
        W(i + l*N_data2,j) = data2[i][j + l*N];

  // store the target pairwise cross-correlation in a matrix
  MatrixXf tildeC(N_data1*nz, N_data1*nz);
  for(i = 0; i < N_data1*nz; i++)
    for(j = 0; j < N_data1*nz; j++)
      tildeC(i,j) = cross_cor[j + i*N_data1*nz];

  // store the target cross-correlation in a matrix
  MatrixXf tildeD(N_data1*nz, N_data2*nz);
  for(i = 0; i < N_data1*nz; i++)
    for(j = 0; j < N_data2*nz; j++)
      tildeD(i,j) = cross_scale_cor[j + i*N_data2*nz];

  // compute the cross-correlations (Line 1)
  MatrixXf C = V*(V.transpose())/N;
  MatrixXf D = V*(W.transpose())/N;

  // matrix computation (Line 2 and Line 3)
  MatrixXf invE = (W*(W.transpose())/N).inverse();
  MatrixXf F = C - D * invE *(D.transpose());
  MatrixXf tildeF = tildeC - tildeD * invE * (tildeD.transpose());

  // eigen decomposition to compute the square root of matrices
  // F = P_in * D_in * P_in^T
  // tildeF = P_out * D_out* P_out^T
  // the following matrices are actually real-valued
  SelfAdjointEigenSolver<MatrixXf> eigensolver_in(F);
  SelfAdjointEigenSolver<MatrixXf> eigensolver_out(tildeF);
  MatrixXcf D_in = eigensolver_in.eigenvalues().asDiagonal();
  MatrixXcf P_in = eigensolver_in.eigenvectors();
  MatrixXcf D_out = eigensolver_out.eigenvalues().asDiagonal();
  MatrixXcf P_out = eigensolver_out.eigenvectors();

  // square root and inverse square root
  // isD_in = D_in^{-1/2}
  // sD_out = D_out^{1/2}
  MatrixXcf isD_in = MatrixXcf::Zero(N_data1*nz,N_data1*nz);
  MatrixXcf sD_out = MatrixXcf::Zero(N_data1*nz,N_data1*nz);
  int test = 0; // to test if the matrix stays at 0 or not
  for(i = 0; i < N_data1*nz; i++) {
    if( fabs(D_in.real()(i,i)) > 1e-12 ) {
      isD_in(i,i) = ((complex<float>) 1)/sqrt((complex<float>) D_in(i,i));
      test = 1;
    }
    sD_out(i,i) = sqrt((complex<float>) D_out(i,i));
  }

  if ( test ) { // if the matrix has non-zero values
    // define new matrix Vnew = Lambda*V + Sigma*Y (Line 4 to Line 6)
    MatrixXcf Lambda = P_out*sD_out*(P_out.transpose())*P_in*isD_in*(P_in.transpose());
    MatrixXcf Sigma = (tildeD.cast<complex<float> >() - Lambda*D.cast<complex<float> >())
                  * invE.cast<complex<float> >();
    MatrixXcf Vnew = Lambda*V.cast<complex<float> >() + Sigma*W.cast<complex<float> >();

    // compute variance of the real and imaginary part
    // done outside compute_moment because of the data structure
    for(l = 0; l < nz; l++) {
      double mr = 0.0;
      double mi = 0.0;
      for(i = 0; i < N_data1; i++)
        for(j = 0; j < N; j++) {
          mr += Vnew.real()(i + l*N_data1,j);
          mi += Vnew.imag()(i + l*N_data1,j);
        }
      mr /= (N_data1*N);
      mi /= (N_data1*N);
      double vr = 0.0;
      double vi = 0.0;
      for(i = 0; i < N_data1; i++)
        for(j = 0; j < N; j++) {
          vr += (Vnew.real()(i + l*N_data1,j) - mr)
            * (Vnew.real()(i + l*N_data1,j) - mr);
          vi += (Vnew.imag()(i + l*N_data1,j) - mi)
            * (Vnew.imag()(i + l*N_data1,j) - mi);
        }

      // do not modify data1 if the variance of the imaginary part is too high
      if(vi/vr < tol) {
        for(i = 0; i < N_data1; i++)
          for(j = 0; j < N; j++)
            data1[i][j + l*N]= Vnew.real()(i + l*N_data1,j);
      }
    }
  }
}

// Adjust the cross-correlations of an image given a list of
// fixed sub-bands (Appendix A.3)
// This corresponds to Algorithm 10
// Linearly adjust variables in data1 to have the variance var
// and the cross-correlation cross_scale_cor
// The input data is assumed to have a zero mean
// Note that data2 is not modified
// The computations are similar to adjust_cross_scale_cor but the data
// has a different structure
void adjust_cross_scale_cor2(float *data1, float **data2,
                          float var, const float *cross_scale_cor,
                          int N_data, int N)
{
  int i, j, l;
  MatrixXf V(1, N);
  MatrixXf W(3*N_data, N);

  // put data in a matrix X
  for(j = 0; j < N; j++)
    V(0,j) = data1[j];

  // put data in a matrix Y
  for(i = 0; i < N_data; i++)
    for(j = 0; j < N; j++)
      for(l = 0; l < 3; l++)
        W(i + l*N_data,j) = data2[i][j + l*N];

  // store the target pairwise cross-correlation in a matrix
  MatrixXf tildeC(1, 1);
  tildeC(0,0) = var;

  // store the target cross-correlation in a matrix
  MatrixXf tildeD(1, 3*N_data);
  for(j = 0; j < 3*N_data; j++)
    tildeD(0,j) = cross_scale_cor[j];

  // compute the cross-correlations (Line 1)
  MatrixXf C = V*(V.transpose())/N;
  MatrixXf D = V*(W.transpose())/N;

  // matrix computation (Line 2 and Line 3)
  MatrixXf invE = (W*(W.transpose())/N).inverse();
  MatrixXf F = C - D * invE * (D.transpose());
  MatrixXf tildeF = tildeC - tildeD * invE * (tildeD.transpose());

  // square root and inverse square root
  // isD_in = F^{-1/2}
  // sD_out = tildeF^{1/2}
  MatrixXcf isD_in = MatrixXcf::Zero(1,1);
  MatrixXcf sD_out = MatrixXcf::Zero(1,1);
  if( fabs(F(0,0) > 1e-12) ) { // only modify if non-zero value
    isD_in(0,0)=((complex<float>) 1)/sqrt((complex<float>) F(0,0));
    sD_out(0,0)=sqrt((complex<float>) tildeF(0,0));

    // define new matrix Vnew = Lambda*V + Sigma*Y (Line 4 to Line 6)
    MatrixXcf Lambda = sD_out*isD_in;
    MatrixXcf Sigma = (tildeD.cast<complex<float> >() - Lambda*D.cast<complex<float> >())
                  * invE.cast<complex<float> >();
    MatrixXcf Vnew = Lambda*V.cast<complex<float> >() + Sigma*W.cast<complex<float> >();

    // compute variance of the real and imaginary part
    // done outside a function because of the data structure
    double mr = 0.0;
    double mi = 0.0;
    for(j=0; j<N; j++) {
      mr += Vnew.real()(0,j);
      mi += Vnew.imag()(0,j);
    }
    mr /= N;
    mi /= N;
    double vr = 0.0;
    double vi = 0.0;
    for(j=0; j<N; j++) {
      vr += (Vnew.real()(0,j)-mr)*(Vnew.real()(0,j)-mr);
      vi += (Vnew.imag()(0,j)-mi)*(Vnew.imag()(0,j)-mi);
    }

    // do not modify data1 if the variance of the imaginary part is too high
    if(vi/vr < 1e-3) {
      for(j=0; j<N; j++)
        data1[j]= Vnew.real()(0,j);
    }
  }
}
