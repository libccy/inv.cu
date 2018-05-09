#!/bin/bash
nvcc inv.cu --std=c++11 -arch=sm_50 -lcublas -lcusolver -lcufft -o inv.out
nvcc util/checker.cu --std=c++11 -arch=sm_50 -lcublas -lcusolver -lcufft -o checker.out
