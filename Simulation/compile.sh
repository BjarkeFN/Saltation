#!/bin/bash
rm branching; g++ -O3 -mavx -std=c++20 $1 -DARMA_DONT_USE_WRAPPER -lblas -llapack -lboost_serialization -lboost_system -lboost_filesystem -o branching;
