// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2021, Thibaud Briand <briand.thibaud@gmail.com>
// Copyright (C) 2021, Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
// All rights reserved.

// Get the value of a pixel array
static float getpixel(const float *x, int w, int h, int i, int j)
{
  if (i < 0 || i >= w || j < 0 || j >= h)
    return 0;

  return x[i + j*w];
}

// Set the value of a pixel array
static void setpixel(float *x, int w, int h, int i, int j, float v)
{
  if (i < 0 || i >= w || j < 0 || j >= h)
    return;

  x[i + j*w] = v;
}

// Evaluate the value of the bilinear zoom
static float evaluate_bilinear(float a, float b, float c, float d, float x,
                               float y)
{
  float r = 0;

  r += a * (1-x) * (1-y);
  r += b * ( x ) * (1-y);
  r += c * (1-x) * ( y );
  r += d * ( x ) * ( y );

  return r;
}

// Compute the bilinear zoom of an image
void zoom_bilinear(float *X, int W, int H, float *x, int w, int h, int pd)
{
  // set ratio of zoom
  float wfactor = w/(float)W;
  float hfactor = h/(float)H;

  float p, q, a, b, c, d, r;
  int ip, iq, i, j, l;

  // loop on the whole output array
  for (l = 0; l < pd; l++) {
    for (j = 0; j < H; j++) {
      q = j*hfactor;
      iq = q;
      for (i = 0; i < W; i++) {
        p = i*wfactor;
        ip = p;

        a = getpixel(x, w, h, ip  , iq  );
        b = getpixel(x, w, h, ip+1, iq  );
        c = getpixel(x, w, h, ip  , iq+1);
        d = getpixel(x, w, h, ip+1, iq+1);
        r = evaluate_bilinear(a, b, c, d, p-ip, q-iq);

        setpixel(X, W, H, i, j, r);
      }
    }

    // go to next channel
    x += w*h;
    X += W*H;
  }
}
