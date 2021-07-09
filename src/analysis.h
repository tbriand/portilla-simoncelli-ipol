// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2021, Thibaud Briand <briand.thibaud@gmail.com>
// Copyright (C) 2021, Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
// All rights reserved.

/* analysis.cpp contains the functions for analyzing an image
 * and computing its statistics
 * This corresponds to Line 1 and Line 2 of Algorithm 5
 */

#ifndef ANALYSISH
#define ANALYSISH

#include "ps_lib.h"

void analysis(statsStruct *stats, imageStruct sample, const paramsStruct params);
void analysis2(statsStruct *stats, imageStruct sample, pyramidStruct pyramid,
               const filtersStruct filters, const paramsStruct params);

#endif
