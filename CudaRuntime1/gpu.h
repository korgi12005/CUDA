#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

inline __global__ void cu_saxpy(float a, float* x, int incx, float* y, int incy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    y[i * incy] = y[i * incy] + a * x[i * incx];
}

inline __global__ void cu_daxpy(double a, double* x, int incx, double* y, int incy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    y[i * incy] = y[i * incy] + a * x[i * incx];

    
}
