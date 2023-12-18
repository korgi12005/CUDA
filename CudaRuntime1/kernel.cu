#include <cstdlib>
#include <iostream>
#include <stdio.h>

#include "omp.h"
#include "gpu.h"
#include "cpu.h"


const  int size = 1000000;
float bigX[size], bigY[size];
double bigXX[size], bigYY[size];

void print(float* arr, int size)
{
    for (int i = 0; i < size; ++i)
        std::cout << arr[i] << ' ';
    std::cout << std::endl;
}
void printD(double* arr, int size)
{
    for (int i = 0; i < size; ++i)
        std::cout << arr[i] << ' ';
    std::cout << std::endl;
}
inline void cuda_s(int block_num, int block_size, size_t n, float a, float* x, int incx, float* y, int incy) {
    printf("block num - %i \n block size - %i \n", block_num, block_size);

    float* dev_x = 0;
    float* dev_y = 0;
    double start=0, end=0;
    cudaSetDevice(0);
    cudaMalloc(&dev_x, n * sizeof(float));
    cudaMalloc(&dev_y, n * sizeof(float));
    cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    start = omp_get_wtime();
    cu_saxpy << <block_num, block_size >> > (a, dev_x, incx, dev_y, incy);
    cudaDeviceSynchronize();
    end = omp_get_wtime();
    printf("time calculate - %f \n", end - start);
    cudaMemcpy(y, dev_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_y);
    cudaFree(dev_x);
}

static void cuda_d(int block_num, int block_size, int n, double a, double* x, int incx, double* y, int incy) {
    double* dev_x = 0;
    double* dev_y = 0;

    double start=0.0, end=0.0;
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "saxpy_gpu launch failed1: %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaStatus = cudaMalloc(&dev_x, n * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "saxpy_gpu launch failed2: %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaStatus = cudaMemcpy(dev_x, x, n * sizeof(double), cudaMemcpyHostToDevice); cudaStatus = cudaMalloc(&dev_y, n * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "saxpy_gpu launch failed4: %s\n", cudaGetErrorString(cudaStatus));
    }
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "saxpy_gpu launch failed3: %s\n", cudaGetErrorString(cudaStatus));
    }
    
    cudaStatus = cudaMemcpy(dev_y, y, n * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "saxpy_gpu launch failed5: %s\n", cudaGetErrorString(cudaStatus));
    }
    start = omp_get_wtime();
    cu_daxpy << <block_num, block_size >> > (a, dev_x, incx, dev_y, incy);
    cudaDeviceSynchronize();
    end = omp_get_wtime();
    printf("time calculate - %f \n", end - start);
    cudaMemcpy(y, dev_y, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dev_y);
    cudaFree(dev_x);

}

void createBigArrF(int size, float* x, float* y) {
    int start = 1;
    int end = 30;
    for (int i = 0; i < size; i++) {
        x[i] = rand() % (end - start + 1) + start;
        y[i] = rand() % (end - start + 1) + start;

    }

}
void createBigArrD(int size, double* x, double* y) {
    int start = 1;
    int end = 30;
    for (int i = 0; i < size; i++) {
        x[i] = rand() % (end - start + 1) + start;
        y[i] = rand() % (end - start + 1) + start;

    }

}
int main()
{
    int n = 5;
    float a = 3;
    double a_d = 3;
    float x_f[5]{ 2.0, 3.0, 6.0, 1.0, 7.0 };
    double x_d[5]{ 2.0, 3.0, 6.0, 1.0, 7.0 };
    int incx = 2;
    float y_f_cpu[5]{ 1.0, 2.0, 3.0 ,4.0 ,5.0 };
    float y_f_gpu[5]{ 1.0, 2.0, 3.0 ,4.0 ,5.0 };
    float y_f_omp[5]{ 1.0, 2.0, 3.0 ,4.0 ,5.0 };
    double y_d_cpu[5]{ 1.0, 2.0, 3.0 ,4.0 ,5.0 };
    double y_d_gpu[5]{ 1.0, 2.0, 3.0 ,4.0 ,5.0 };
    double y_d_omp[5]{ 1.0, 2.0, 3.0 ,4.0 ,5.0 };
    int incy = 2;
    int block_sizes = 4;
    int blocks_num = n / block_sizes;
    double start, end;
    printf("Float calulate \n");
    printf("cpu calc\n");
    saxpy(3, a, x_f, incx, y_f_cpu, incy);
    print(y_f_cpu, n);
    printf("gpu calc\n");
    cuda_s(blocks_num, block_sizes, n,a,x_f,incx, y_f_gpu,incy);
    print(y_f_gpu, n);
    printf("omp calc\n");
    omp_saxpy(n, a, x_f, incx, y_f_omp, incy);
    print(y_f_omp, n);

    printf("Double calulate \n");
    printf("cpu calc\n");
    daxpy(3, a_d, x_d, incx, y_d_cpu, incy);
    printD(y_d_cpu, n);
    printf("gpu calc\n");
    cuda_d(blocks_num, block_sizes, n, a_d, x_d, incx, y_d_gpu, incy);
    printD(y_d_gpu, n);
    printf("omp calc\n");
    omp_daxpy(n, a_d, x_d, incx, y_d_omp, incy);
    printD(y_d_omp, n);

    const int block_sizes_arr[] = { 8, 16, 32, 64, 128, 256 };
    for (int i = 0; i < 6; i++) {
        createBigArrD(size, bigXX, bigYY);
        long blocks_num = size / block_sizes_arr[i];
        printf("test gpu on block size = %i \n", block_sizes_arr[i]);
        cuda_d(blocks_num, block_sizes_arr[i], size, a_d, bigXX, incx, bigYY, incy);
    }


    blocks_num = size / block_sizes;
    createBigArrF(size, bigX, bigY);
    printf("cpu test big array \n");

    start = omp_get_wtime();
    saxpy(size/incx, a, bigX, incx, bigY, incy);
    end = omp_get_wtime();
    printf("time calculate - %f \n", end - start);
    createBigArrF(size, bigX, bigY);

    printf("gpu test big array \n");
    cuda_s(blocks_num, block_sizes, size, a, bigX, incx, bigY, incy);
    
    createBigArrF(size, bigX, bigY);
    printf("omp test big array\n");
    start = omp_get_wtime();
    omp_saxpy(size/incx, a, bigX, incx, bigY, incy);
    end = omp_get_wtime();
    printf("time calculate - %f \n", end - start);


   

    //cudaMalloc((void**)&a_gpu, n * sizeof(int));
    //cudaMemcpy(a_gpu, a, n * sizeof(int), cudaMemcpyHostToDevice);

    //second_task << <num_block, block_size >> > (a_gpu, n);
    //cudaMemcpy(a, a_gpu, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    //cudaFree(a_gpu);

    cudaDeviceSynchronize();
    return 0;
}