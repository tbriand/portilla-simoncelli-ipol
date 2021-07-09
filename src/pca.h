// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2021, Thibaud Briand <briand.thibaud@gmail.com>
// Copyright (C) 2021, Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
// All rights reserved.

/* pca.cpp contains the auxiliary functions for applying the direct and
 * indirect PCA transform to color images (Appendix B.1)
 */

#ifndef PCAH
#define PCAH

void compute_covariance(float *im, float C[3][3], int N);
void eigen_decomposition(const float A[3][3], float V[3][3], float d[3]);
void apply_pca(float *out, const float *in, const float V[3][3],
               const float d[3], int N);
void apply_inverse_pca(float *out, const float *in, const float V[3][3],
                       const float d[3], int N);

#endif
