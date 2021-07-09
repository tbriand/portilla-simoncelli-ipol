// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2021, Thibaud Briand <briand.thibaud@gmail.com>
// Copyright (C) 2021, Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
// All rights reserved.

/* synthesis.cpp contains the functions for the iterative synthesis of the
 * texture given the summary statistics (Algorithm 4 and Line 3 to Line 9 of Algorithm 5)
 */

#ifndef SYNTHESISH
#define SYNTHESISH

#include "ps_lib.h"

void synthesis(imageStruct texture, const statsStruct stats,
               const paramsStruct params);

#endif
