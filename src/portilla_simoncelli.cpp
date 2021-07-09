// This program is free software: you can use, modify and/or redistribute it
// under the terms of the simplified BSD License. You should have received a
// copy of this license along this program. If not, see
// <http://www.opensource.org/licenses/bsd-license.html>.
//
// Copyright (C) 2021, Thibaud Briand <briand.thibaud@gmail.com>
// Copyright (C) 2021, Jonathan Vacher <jonathan.vacher@einstein.yu.edu>
// All rights reserved.

/* See Section C.1 for more details about input handling and options */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern "C" {
#include "iio.h"
}
#include "ps_lib.h"
#include "mt19937ar.h"
#include "constraints.h"
#include "pca.h"

// default parameters
#define PAR_DEFAULT_NSTEER 4
#define PAR_DEFAULT_NPYR 4
#define PAR_DEFAULT_NITERATION 50
#define PAR_DEFAULT_NA 7
#define PAR_DEFAULT_NX 0
#define PAR_DEFAULT_NY 0
#define PAR_DEFAULT_EDGEHANDLING 0
#define PAR_DEFAULT_ADD_SMOOTH 0
#define PAR_DEFAULT_CROP 0
#define PAR_DEFAULT_SEED 0
#define PAR_DEFAULT_NOISE 0
#define PAR_DEFAULT_MARG_STATS 1
#define PAR_DEFAULT_AUTOCORR 1
#define PAR_DEFAULT_MAGCORR 1
#define PAR_DEFAULT_REALCORR 1
#define PAR_DEFAULT_VERBOSE 0
#define PAR_DEFAULT_INTERPWEIGHT 0.5
#define PAR_DEFAULT_GRAY 0
#define PAR_DEFAULT_STATISTICS 0

// print a message and abort the execution
#define FATAL(MSG)                      \
do {                                    \
        fprintf(stderr, MSG "\n");      \
        abort();                        \
   } while (0);

// Display help usage
static void print_help(char *name)
{
  printf("\n<Usage>: %s input output [OPTIONS] \n\n", name);

  printf("The optional parameters are:\n");
  printf("-s, \t Specify the number of scales P (by default %i)\n", PAR_DEFAULT_NPYR);
  printf("-k, \t Specify the number of orientations (by default %i)\n", PAR_DEFAULT_NSTEER);
  printf("-i, \t Specify the number of iterations (by default %i)\n", PAR_DEFAULT_NITERATION);
  printf("-N, \t Specify the size Na of the neighborhood for the auto-correlation adjustment (by default %i)\n", PAR_DEFAULT_NA);
  printf("-v, \t Activate verbose mode\n");

  printf("\n");
  printf("Size parameters:\n");
  printf("-x, \t Specify the output width (by default the input width)\n");
  printf("-y, \t Specify the output height (by default the input height)\n");
  printf("The output size is required to be a multiple of 2^(P + 1). Otherwise a crop is performed\n");
  printf("-c, \t Save the cropped input (1) or not (0) (default %i)\n", PAR_DEFAULT_CROP);

  printf("\n");
  printf("Initial noise parameters:\n");
  printf("-g, \t Specify the noise seed (by default %i)\n", PAR_DEFAULT_SEED);
  printf("-n, \t Filename of the initial noise\n");

  printf("\n");
  printf("Adjusment of the constraints parameters:\n");
  printf("-C1, \t Adjust marginal statistics (1) or not (0) (by default %i)\n", PAR_DEFAULT_MARG_STATS);
  printf("-C2, \t Adjust auto-correlation (1) or not (0) (by default %i)\n", PAR_DEFAULT_AUTOCORR);
  printf("-C3, \t Adjust magnitude correlation (1) or not (0) (by default %i)\n", PAR_DEFAULT_MAGCORR);
  printf("-C4, \t Adjust real correlation (1) or not (0) (by default %i)\n", PAR_DEFAULT_REALCORR);
  printf("-o, \t Write the statistics evolution in a file (1) or not (0) (by default %i)\n", PAR_DEFAULT_STATISTICS);

  printf("\n");
  printf("Interpolation of two input textures:\n");
  printf("-t, \t Filename of the second input texture\n");
  printf("-w, \t Specify the weight of the first texture (by default %f)\n", PAR_DEFAULT_INTERPWEIGHT);

  printf("\n");
  printf("Edge handling parameters:\n");
  printf("-e, \t Use periodic plus smooth decomposition (1) or not (0) (default %i)\n", PAR_DEFAULT_EDGEHANDLING);
  printf("-a, \t Add the smooth component (1) or not (0) (default %i)\n", PAR_DEFAULT_ADD_SMOOTH);

  printf("\n");
  printf("Grayscale conversion (average of the channels):\n");
  printf("-b, \t Use grayscale mode (1) or not (0) (default %i)\n", PAR_DEFAULT_GRAY);
}

// read command line parameters
static int read_parameters(int argc, char *argv[], char **infile,
                           char **outfile, int &N_steer, int &N_pyr,
                           int &N_iteration, int &Na, int &noise,
                           char **noisefile, int &nx, int &ny,
                           int &edge_handling, int &add_smooth, int &crop,
                           unsigned long &seed, int cmask[4], int &verbose,
                           char **infile2, float &interpWeight, int &gray,
                           int &statistics)
{
  // display usage
  if (argc < 3) {
    print_help(argv[0]);
    return 0;
  }
  else {
    int i = 1;
    *infile  = argv[i++];
    *outfile = argv[i++];

    // default value initialization
    N_pyr         = PAR_DEFAULT_NPYR;
    N_steer       = PAR_DEFAULT_NSTEER;
    N_iteration   = PAR_DEFAULT_NITERATION;
    Na            = PAR_DEFAULT_NA;
    verbose       = PAR_DEFAULT_VERBOSE;
    edge_handling = PAR_DEFAULT_EDGEHANDLING;
    add_smooth    = PAR_DEFAULT_ADD_SMOOTH;
    crop          = PAR_DEFAULT_CROP;
    nx            = PAR_DEFAULT_NX;
    ny            = PAR_DEFAULT_NY;
    seed          = PAR_DEFAULT_SEED;
    noise         = PAR_DEFAULT_NOISE;
    cmask[0]      = PAR_DEFAULT_MARG_STATS;
    cmask[1]      = PAR_DEFAULT_AUTOCORR;
    cmask[2]      = PAR_DEFAULT_MAGCORR;
    cmask[3]      = PAR_DEFAULT_REALCORR;
    gray          = PAR_DEFAULT_GRAY;
    statistics   = PAR_DEFAULT_STATISTICS;
    int interpolate = 0;

    // read each parameter from the command line
    while(i < argc) {
      if(strcmp(argv[i],"-s")==0)
        if(i < argc-1)
          N_pyr = atoi(argv[++i]);

      if(strcmp(argv[i],"-k")==0)
        if(i < argc-1)
          N_steer = atoi(argv[++i]);

      if(strcmp(argv[i],"-i")==0)
        if(i < argc-1)
          N_iteration = atoi(argv[++i]);

      if(strcmp(argv[i],"-N")==0)
        if(i < argc-1)
          Na = atoi(argv[++i]);

      if(strcmp(argv[i],"-v")==0)
        verbose = 1;

      if(strcmp(argv[i],"-e")==0)
        if(i < argc-1)
          edge_handling = atoi(argv[++i]);

      if(strcmp(argv[i],"-a")==0)
        if(i < argc-1)
          add_smooth=atoi(argv[++i]);

      if(strcmp(argv[i],"-c")==0)
        if(i < argc-1)
          crop=atoi(argv[++i]);

      if(strcmp(argv[i],"-x")==0)
        if(i < argc-1)
          nx = atoi(argv[++i]);

      if(strcmp(argv[i],"-y")==0)
        if(i < argc-1)
          ny = atoi(argv[++i]);

      if(strcmp(argv[i],"-g")==0)
        if(i < argc-1)
          seed = atoi(argv[++i]);

      if(strcmp(argv[i],"-n")==0)
        if(i < argc-1) {
          // noise specified
          noise = 1;
          *noisefile = argv[++i];
      }

      if(strcmp(argv[i],"-C1")==0)
        if(i < argc-1)
          cmask[0] = atoi(argv[++i]);

      if(strcmp(argv[i],"-C2")==0)
        if(i < argc-1)
          cmask[1] = atoi(argv[++i]);

      if(strcmp(argv[i],"-C3")==0)
        if(i < argc-1)
          cmask[2] = atoi(argv[++i]);

      if(strcmp(argv[i],"-C4")==0)
        if(i < argc-1)
          cmask[3] = atoi(argv[++i]);

      if(strcmp(argv[i],"-t")==0)
        if(i < argc-1) {
          // second texture specified
          if (interpWeight < 0)
            interpWeight  = PAR_DEFAULT_INTERPWEIGHT;
          interpolate = 1;
          *infile2 = argv[++i];
        }

      if(strcmp(argv[i],"-w")==0)
        if(i < argc-1)
          interpWeight = atof(argv[++i]);

      if(strcmp(argv[i],"-b")==0)
        if(i < argc-1)
          gray = atoi(argv[++i]);

      if(strcmp(argv[i],"-o")==0)
        if(i < argc-1)
          statistics = atoi(argv[++i]);

      i++;
    }

    // check consistency
    if ( add_smooth == 1 && edge_handling != 1) {
      if ( verbose )
        printf("Cannot add the smooth component when the periodic plus smooth decomposition is not used\n");
      add_smooth = 0;
    }

    if ( interpolate == 0 && interpWeight >= 0) {
      if ( verbose )
        printf("Second input texture is missing\n");
      interpWeight = -1;
    }

    if ( N_steer < 3 )
      if ( verbose )
        printf("The number of orientations must be higher than 3. Setting the value to %i\n", PAR_DEFAULT_NSTEER);

    if (interpWeight > 1) {
      if ( verbose )
        printf("Interpolation weight cannot be higher than 1\n");
      interpWeight = 1;
    }

    return 1;
  }
}

// Crop the input image to be able to compute the pyramid decomposition
// The sizes must be multiples of 2^{N_pyr + 1}
void perform_crop(const imageStruct data, imageStruct *data_crop, int N_pyr) {
  // number of channels
  int nz = data_crop->nz = data.nz;

  // compute 2^{N_pyr + 1}
  int power_N_pyr = 1<<(N_pyr+1);

  // horizontal
  int rem_nx = data.nx % power_N_pyr; // nx = 2^(N_pyr + 1)*q + rem_nx
  int r_nx = (rem_nx/2);
  int nx_crop = data_crop->nx = data.nx - rem_nx;

  // vertical
  int rem_ny = data.ny % power_N_pyr; // ny = 2^(N_pyr + 1)*q + rem_ny
  int r_ny = (rem_ny/2);
  int ny_crop = data_crop->ny = data.ny - rem_ny;

  // memory allocation
  data_crop->image = (float *) malloc(nz * nx_crop * ny_crop * sizeof(float));

  // compute the cropped image
  for(int l = 0; l < nz; l++)
    for(int j = 0; j < ny_crop; j++)
      for (int i = 0; i < nx_crop; i++)
        data_crop->image[i + j*nx_crop + l*nx_crop*ny_crop] =
          data.image[i + r_nx + (j + r_ny)*data.nx + l*data.nx*data.ny];
}

// Conversion from color to grayscale
static void grayscale(imageStruct data) {
  float factor = 1.0/3;
  int nx = data.nx, ny = data.ny;
  for(int i = 0; i < nx*ny; i++)
    data.image[i] = (data.image[i] + data.image[i + nx*ny]
      + data.image[i + 2*nx*ny])*factor;
}

// Sanity check to avoid odd cases with constant image:
// 1) For color images: If the color covariance matrix has an eigenvalue
//    too small then the color input image is replaced by the grayscale
//    average of its channels.
// 2) For grayscale images: checks if the image is close to a constant
//    by computing its variance
static int sanity_check(imageStruct *data) {
  int i, l;
  float m;
  int nx = data->nx;
  int ny = data->ny;
  int nz = data->nz;

  // handling color images
  if( nz == 3) {
    // memory allocation
    float *tmp = (float *) malloc(nx*ny*nz*sizeof(float));

    // substract the mean value
    for(l = 0; l < 3; l++) {
      m = mean(data->image + l*nx*ny, nx*ny);
      for(i = 0; i < nx*ny; i++)
        tmp[i + l*nx*ny] = data->image[i + l*nx*ny] - m;
    }

    // compute the color covariance matrix
    float covariance[3][3];
    compute_covariance(tmp, covariance, nx*ny);

    // compute the eigenvector decomposition of the color covariance matrix
    float eigenVectors[3][3];
    float eigenValues[3];
    eigen_decomposition(covariance, eigenVectors, eigenValues);

    // change to grayscale mode if a PCA band is too close to a constant
    int test = 0;
    for(l = 0; l < 3; l++)
      if ( eigenValues[l] < 1e-2 )
        test = 1;
    if ( test ) {
      data->nz = nz = 1;
      grayscale(*data);
    }

    // free memory
    free(tmp);
  }

  // determine if the input image is too close to a constant
  int not_constant = 1;
  if ( nz == 1 ) { // test already done for color images
    m = mean(data->image, nx*ny);
    float var = compute_moment(data->image, m, 2, nx*ny);
    if ( var < 1e-2 )
      not_constant = 0;
  }

  return not_constant;
}

// Main function call (input-output and options handling)
int main(int argc, char **argv)
{
  // parameters declaration
  char *infile, *outfile, *noisefile = NULL, *infile2 = NULL;
  int N_steer, N_pyr, N_iteration, Na, verbose = 0;
  int nxout = 0, nyout = 0, noise, edge_handling, add_smooth, crop;
  int gray, statistics;
  int cmask[4] = { 0 };
  unsigned long seed;
  float interpWeight = -1; // negative value <--> no interpolation

  // read parameters
  int result = read_parameters(
    argc, argv, &infile, &outfile, N_steer, N_pyr, N_iteration,
    Na, noise, &noisefile, nxout, nyout, edge_handling,
    add_smooth, crop, seed, cmask, verbose,
    &infile2, interpWeight, gray, statistics);

  if( result ) { // if the parameters are correct
    // read input image
    imageStruct data_in;
    data_in.image = iio_read_image_float_split(infile, &data_in.nx, &data_in.ny,
                                               &data_in.nz);
    int nz = data_in.nz;

    // adjust number of channels (to avoid odd cases or to use grayscale mode)
    nz = (nz > 3) ? 3 : nz;
    nz = (nz == 2) ? 1 : nz;
    if ( gray && nz == 3) {
      if ( verbose )
        printf("Using the grayscale mode\n");
      grayscale(data_in);
      nz = 1;
    }
    data_in.nz = nz;

    // sanity check for image content (possibly modify the number of channels)
    if ( verbose )
      printf("Reading the first input texture\n");
    if ( sanity_check(&data_in) ) { // the image is not constant
      if ( verbose && nz != data_in.nz)
        printf("Taking the grayscale average of the input color texture\n");
      nz = data_in.nz; // the value could have been modified in sanity_check

      // read second image if provided
      imageStruct data_in2;
      if ( interpWeight >= 0 ) {
        data_in2.image = iio_read_image_float_split(infile2, &data_in2.nx,
                                                    &data_in2.ny, &data_in2.nz);
        int nz2 = data_in2.nz;

        // adjust number of channels (avoid odd cases)
        nz2 = (nz2 > 3) ? 3 : nz2;
        nz2 = (nz2 == 2) ? 1 : nz2;
        if ( gray && nz2 == 3) {
          grayscale(data_in2);
          nz2 = 1;
        }
        data_in2.nz = nz2;

        // sanity check for second input texture
        if (verbose)
          printf("Reading the second input texture\n");
        if ( sanity_check(&data_in2) ) {
          if ( verbose && nz2 != data_in2.nz)
            printf("Taking the grayscale average of the second input color texture\n");
          // adjust the number of channels to the same value
          if ( nz != data_in2.nz)
            nz = data_in.nz = data_in2.nz = 1;
        }
        else {
          if ( verbose )
            printf("Second input texture is too close to a constant and is not used\n");
          interpWeight = -1;
        }
      }

      // change the number of scales N_pyr if too high
      // the input sizes after cropping must be greater than 2^(N_pyr+1)*Na
      // this is guaranteed as long as the sizes before cropping are
      // greater than 2^(N_pyr+1)*(Na+1)
      int min_size = (data_in.nx > data_in.ny) ? data_in.ny : data_in.nx;
      if ( interpWeight >= 0) { // take the min over the two input images
        int min_size2 = (data_in2.nx > data_in2.ny) ? data_in2.ny : data_in2.nx;
        min_size = (min_size > min_size2) ? min_size2 : min_size;
      }
      int N_pyr_max = (log(min_size)-log(Na+1))/log(2)-1;
      if ( N_pyr_max <= 0 )
        FATAL("Input sample is too small for this value of Na");
      if (N_pyr > N_pyr_max) {
        N_pyr = N_pyr_max;
        if ( verbose )
          printf("Input sample is too small for this value of N_pyr. Changing N_pyr to %i\n", N_pyr);
      }

      // crop the image if necessary
      imageStruct data_in_crop;
      perform_crop(data_in, &data_in_crop, N_pyr);
      free(data_in.image);

      imageStruct data_in2_crop;
      if ( interpWeight >= 0 ) {
        perform_crop(data_in2, &data_in2_crop, N_pyr);
        free(data_in2.image);
      }

      // output initialization
      imageStruct noise_in, data_out;
      if ( noise ) { // if the noise file is specified then initialize with it
        if (verbose)
          printf("Reading the input noise\n");
        noise_in.image = iio_read_image_float_split(noisefile, &noise_in.nx,
                                                    &noise_in.ny, &noise_in.nz);
        int nz2 = noise_in.nz;
        // adjust number of channels (avoid odd cases)
        nz2 = (nz2 > 3) ? 3 : nz2;
        nz = (nz == 2) ? 1 : nz2;
        if ( gray && nz2 == 3) {
          grayscale(noise_in);
          nz2 = 1;
        }
        noise_in.nz = nz2;

        // check sizes and content
        if ( noise_in.nx < (1<<(N_pyr+1))*(Na + 1) || noise_in.ny < (1<<(N_pyr+1))*(Na + 1) ) {
          if (verbose) {
            printf("The input noise is too small for Na=%i and N_pyr=%i\n", Na, N_pyr);
            printf("Switching to random noise initialization");
          }
          noise = 0;
        }
        else if ( sanity_check(&noise_in) ) { // if the input noise is not constant
          if ( verbose && nz2 != noise_in.nz)
            printf("Taking the grayscale average of the input color noise\n");
          if ( noise_in.nz != nz ) {
            if (verbose) {
              printf("The input noise should have the same number of channels as the input sample\n");
              printf("Switching to grayscale mode\n");
            }
            nz = data_in_crop.nz = data_in2_crop.nz = noise_in.nz = 1;
          }
          perform_crop(noise_in, &data_out, N_pyr);
        }
        else {
          if (verbose)
            printf("The input noise is too close to a constant and is not used\n");
          noise = 0;
        }
        // free memory
        free(noise_in.image);
      }
      if ( !noise ) { // otherwise set the output sizes and allocate memory
        // the variable noise may be modified in the previous "if" structure
        // so that another "if" structure has to be used (and not an "else" structure)
        nxout = nxout - nxout % (1<<(N_pyr+1));
        if ( nxout < (1<<(N_pyr+1))*Na ) {
          if ( verbose )
            printf("Setting the output width to the input one\n");
          nxout = data_in_crop.nx;
        }
        nyout = nyout - nyout % (1<<(N_pyr+1));
        if ( nyout < (1<<(N_pyr+1))*Na ) {
          if ( verbose )
            printf("Setting the output height to the input one\n");
          nyout = data_in_crop.ny;
        }
        data_out.nx = nxout;
        data_out.ny = nyout;
        data_out.nz = nz;
        data_out.image = (float *) malloc(nz * nxout * nyout * sizeof(float));
      }

      // write the cropped image if asked
      if( crop ) {
        iio_write_image_float_split( (char*) "input_cropped.png",
          data_in_crop.image, data_in_crop.nx, data_in_crop.ny, nz);
        if (interpWeight >= 0)
          iio_write_image_float_split( (char*) "input2_cropped.png",
            data_in2_crop.image, data_in2_crop.nx, data_in2_crop.ny, nz);
      }

      // create the file for writing the statistics evolution
      if ( statistics ) {
        FILE *fp = fopen("statistics_evolution.txt", "w");
        fclose(fp);
      }

      // store parameters in the structure
      paramsStruct params = { N_steer, N_pyr, N_iteration, Na, noise,
                              edge_handling, add_smooth, {0}, verbose,
                              interpWeight, statistics
                            };
      memcpy(params.cmask, cmask, 4*sizeof(int));

      // initialization of the noise with the provided seed
      mt_init_genrand(seed);

      // Portilla and Simoncelli texture synthesis (Algorithm 5)
      if ( verbose )
        printf("Starting texture synthesis of size (%i,%i,%i)\n",
          data_out.nx, data_out.ny, nz);
      ps(data_out, data_in_crop, data_in2_crop, params);

      // write output
      iio_write_image_float_split(outfile, data_out.image, data_out.nx,
                                  data_out.ny, data_out.nz);

      // free memory
      free(data_in_crop.image);
      if ( interpWeight >= 0 )
        free(data_in2_crop.image);
      free(data_out.image);
    }
    else { // the input image is too close to a constant
      if ( verbose )
        printf("Input image is too close to a constant. No computation was made\n");

      // write output
      iio_write_image_float_split(outfile, data_in.image, data_in.nx,
                                  data_in.ny, data_in.nz);

      // free memory
      free(data_in.image);
    }
  }

  return EXIT_SUCCESS;
}
