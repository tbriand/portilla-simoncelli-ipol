// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2021, Thibaud Briand <briand.thibaud@gmail.com>
// Copyright (C) 2021, Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
// All rights reserved.

/* filters.c contains the functions for computing the filters of the pyramid
 * decomposition (Section 2.1)
 */

#ifndef FILTERS_H
#define FILTERS_H

#include "ps_lib.h"

void compute_filters(filtersStruct *filters, int nx, int ny,
                     int N_pyr, int N_steer, int option);
void free_filters(filtersStruct filters, int N_pyr, int N_steer, int option);

#endif
