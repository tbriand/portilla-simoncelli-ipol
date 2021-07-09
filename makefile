CFLAGS=-Wall -O3 -fopenmp -march=native
LDFLAGS=-lm -lpng -ljpeg -ltiff -lstdc++ -lfftw3f -lfftw3f_omp

SRC1 := src/portilla_simoncelli.cpp src/external/mt19937ar.cpp src/ps_lib.cpp src/analysis.cpp src/synthesis.cpp src/filters.cpp src/toolbox.cpp src/constraints.cpp src/periodic_plus_smooth.cpp src/zoom_bilinear.cpp src/pca.cpp src/pyramid.cpp
SRC2 := src/external/iio.c

INCLUDE = -I./src/Eigen_library -I./src/external

#Replace suffix .cpp and .c by .o
OBJ := $(addsuffix .o,$(basename $(SRC1))) $(addsuffix .o,$(basename $(SRC2)))

#Binary file
BIN := portilla_simoncelli

#All is the target (you would run make all from the command line). 'all' is dependent
#all: $(BIN)

#Generate executables
portilla_simoncelli: $(OBJ)
	$(CXX) -std=c++11 $^ -o $@ $(CFLAGS) $(LDFLAGS)

#each object file is dependent on its source file, and whenever make needs to create
#an object file, to follow this rule:
%.o: %.c
	$(CC) -std=c99  -c $< -o $@ $(INCLUDE) $(CFLAGS)  -Wno-unused -pedantic -DNDEBUG -D_GNU_SOURCE

%.o: %.cpp
	$(CXX) -std=c++11 -c $< -o $@ $(INCLUDE) $(CFLAGS)

clean:
	rm -f $(OBJ) $(BIN)
