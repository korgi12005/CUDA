#pragma once
#include <omp.h>
#include <cassert>

void omp_saxpy(int n, float a, float* x, int incx, float* y, int incy)
{
#pragma omp parallel for
	for (int i = 0; i < n; ++i)
	{
		y[i * incy] = y[i * incy] + a * x[i * incx];
	}
}

inline void omp_daxpy(int n, double a, double* x, int incx, double* y, int incy)
{
#pragma omp parallel for
	for (int i = 0; i < n; ++i)
	{
		y[i * incy] = y[i * incy] + a * x[i * incx];
	}
}