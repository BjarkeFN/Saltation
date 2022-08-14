#!/bin/bash
rm branching; g++-12 -O3 -mavx -std=c++17 $1 -DARMA_DONT_USE_WRAPPER -lblas -llapack -lboost_serialization -lboost_system -lboost_filesystem -o branching;
#rm branching; g++-10 -O3 -mavx -std=c++17 $1 -DARMA_DONT_USE_WRAPPER -lblas -llapack -lboost_serialization -lboost_system -lboost_filesystem -o branching;
