example:
	g++ -o ../example.out  gcn.cpp
example_openmp:
# 	g++ -fopenmp -o ../example_openmp.out  gcn_openmp.cpp
	g++ -fopenmp -O3 -mavx2 -mfma -ffast-math -o ../example_openmp.out gcn_openmp.cpp
all: example example_openmp