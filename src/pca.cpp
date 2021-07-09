// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2021, Thibaud Briand <briand.thibaud@gmail.com>
// Copyright (C) 2021, Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
// All rights reserved.

#include <stdlib.h>
#include <math.h>
#include "Eigen/Dense"
#include <Eigen/Eigenvalues>

using namespace Eigen;
using namespace std;

// Compute the color covariance matrix of a color image (Equation 97)
// The mean of each channel of the input is assumed to be 0
void compute_covariance(float *im, float C[3][3], int N)
{
  int i, l;
  MatrixXf U(3,N);

  // put input data in a matrix
  for(l = 0; l < 3; l++)
    for(i = 0; i < N; i++)
      U(l,i) = im[i + l*N];

  // compute the color covariance matrix (Equation 97)
  Matrix3f cov = U*(U.transpose())/N;

  // put the covariance matrix in 3x3 array
  for(i = 0; i < 3; i++)
    for(l = 0; l < 3; l++)
      C[i][l] = cov(i,l);
}

// Compute the eigenvector decomposition of a symmetric matrix
// We have A = V*diag(d)*V' where:
// 1) V is an orthogonal matrix containing the eigenvectors
// 2) d contains the eigenvalues
void eigen_decomposition(const float A[3][3], float V[3][3], float d[3])
{
  int i, j;
  Matrix3f C;

  // put array A in a matrix C
  for(i = 0; i < 3; i++)
    for(j = 0; j < 3; j++)
      C(i,j) = A[i][j];

  // eigen solver
  SelfAdjointEigenSolver<Matrix3f> eigensolver(C);

  // eigenvectors
  Matrix3f P = eigensolver.eigenvectors();
  for(i = 0; i < 3; i++)
    for(j = 0; j < 3; j++)
      V[i][j] = P(i,j);

  // eigenvalues
  Vector3f E = eigensolver.eigenvalues();
  for(i = 0; i < 3; i++)
    d[i] = E(i);
}

// Apply the direct PCA transform (Equation 99)
// out = diag(d)^{-1/2}*V'*in
// The mean of each channel of the input is assumed to be 0
void apply_pca(float *out, const float *in, const float V[3][3],
               const float d[3], int N)
{
  int i, l;

  // put data in matrices
  MatrixXf U(3,N);
  for(l = 0; l < 3; l++)
    for(i = 0; i < N; i++)
      U(l,i) = in[i + l*N];

  Matrix3f V2;
  for(l = 0; l < 3; l++)
    for(i = 0; i < 3; i++)
      V2(i,l) = V[i][l];

  // compute d^{-1/2}
  // eigenvalues smaller than 1e-2 are set to 0 to avoid instability
  // this is useless when the function sanity_check was used to handle
  // input images with strange content
  Vector3f d2;
  for(l = 0; l < 3; l++)
    d2(l) = (d[l] < 1e-2) ? 0 : 1.0/sqrt(d[l]);

  // apply direct PCA transform (Equation 82)
  // out = diag(d)^{-1/2}*V'*in
  U = (d2.asDiagonal())*(V2.transpose())*U;

  // put the computed matrix in the output array
  for(l = 0; l < 3; l++)
    for(i = 0; i < N; i++)
      out[i + l*N] = U(l,i);
}

// Apply the inverse PCA transform (Equation 100)
// out = V*diag(d)^{1/2}*in
// The mean of each channel of the input is assumed to be 0
void apply_inverse_pca(float *out, const float *in, const float V[3][3],
                       const float d[3], int N)
{
  int i, l;

  // put data in matrices
  MatrixXf tildeU(3,N);
  for(l = 0; l < 3; l++)
    for(i = 0; i < N; i++)
      tildeU(l,i) = in[i + l*N];

  Matrix3f V2;
  for(l = 0; l < 3; l++)
    for(i = 0; i < 3; i++)
      V2(i,l) = V[i][l];

  // compute d^{1/2}
  Vector3f d2;
  for(l = 0; l < 3; l++)
    d2(l) = sqrt(d[l]);

  // apply inverse PCA transform (Equation 83)
  // out = V*diag(d)^{1/2}*in
  tildeU = V2*(d2.asDiagonal())*tildeU;

  // put the computed matrix in the output array
  for(l = 0; l < 3; l++)
    for(i = 0; i < N; i++)
      out[i + l*N] = tildeU(l,i);
}
