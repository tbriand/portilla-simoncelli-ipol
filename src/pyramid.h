// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2021, Thibaud Briand <briand.thibaud@gmail.com>
// Copyright (C) 2021, Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
// All rights reserved.

/* pyramid.cpp contains the functions for computing the pyramid decomposition
 * of an image (Algorithm 2)
 */

#ifndef PYRAMIDH
#define PYRAMIDH

#include "ps_lib.h"

void create_pyramid(pyramidStruct pyramid, const imageStruct image,
                    const filtersStruct filters, const paramsStruct params,
                    int option);
void allocate_pyramid(pyramidStruct *pyramid, int N_pyr, int N_steer,
                      int nx, int ny, int nz, int option);
void free_pyramid(pyramidStruct pyramid, int N_pyr, int N_steer, int option);

#endif
