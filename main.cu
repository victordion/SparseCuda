#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include <ctime>
#define DEBUG
using namespace std;

struct SubBlock{
    
    int * nnz_global_i_idx;
    int * nnz_global_o_idx;

    int nnz;
    int * nnz_local_r_idx;
    int * nnz_local_c_idx;
    float * nnz_values;
};

__global__ void CudaCompute(SubBlock * d_sbs, float * d_x, float * d_y, int nblocks, int mem_b_size, int nrows, int ncols , float * sub_y_arr){
        /*
            sub_y_arr stores float number, with nblocks rows, mem_b_size columns
        */
        
        if(blockIdx.x >= nblocks)
            return;

        //printf("This is Cuda Block # %d: \n", blockIdx.x);
        SubBlock * work_sb = &d_sbs[blockIdx.x];

        float * x_sub = (float *) malloc(mem_b_size * sizeof(float));
        float * y_sub = (float *) malloc(mem_b_size * sizeof(float));

        for(int i = 0; i < mem_b_size; i++){
            if(work_sb->nnz_global_i_idx[i] > 0 && work_sb->nnz_global_i_idx[i] <= ncols){
                // d_x   indexing starts from '1'
                // x_sub indexing starts from '0'
                x_sub[i] = d_x[work_sb->nnz_global_i_idx[i] - 1];
            }
            else{
                x_sub[i] = 0.0;
            }            
        }

        for(int i = 0; i < work_sb->nnz; i++){
            y_sub[work_sb->nnz_local_r_idx[i] - 1] += work_sb->nnz_values[i] * x_sub[work_sb->nnz_local_c_idx[i] - 1];
        }

        for(int i = 0; i < mem_b_size; i++){
            sub_y_arr[blockIdx.x * mem_b_size + i] = y_sub[i];
        }

}

__global__ void CudaMergeResults(SubBlock * d_sbs, float * d_x, float * d_y, int nblocks, int mem_b_size, int nrows, int ncols , float * sub_y_arr){

    for(int i = 0; i < nblocks; i++){
        int * outLocs = d_sbs[i].nnz_global_o_idx;
        for(int j = 0; j < mem_b_size; j++){
            
            d_y[outLocs[j] - 1] += sub_y_arr[i * mem_b_size + j];
        }
    }
}

void printSubBlocksInfo(SubBlock * sbs, int nsbs, int mem_b_size){
    cout << endl << "There are " << nsbs << "subblocks." << endl;
    for(int i = 0; i < nsbs; i++){
        cout << "Subblock #: " << i << endl;
        cout << "Numbe of non-zeros: " << sbs[i].nnz << endl;

        cout << "Input Locations: " << endl;
        for(int j = 0; j < mem_b_size; j++){
            cout << sbs[i].nnz_global_i_idx[j] << " ";
        }
        cout << endl;

        cout << "Output Locations: " << endl;
        for(int j = 0; j < mem_b_size; j++){
            cout << sbs[i].nnz_global_o_idx[j] << " ";
        }
        cout << endl;
        
        for(int j = 0; j < sbs[i].nnz; j++){
            cout << sbs[i].nnz_local_r_idx[j] << " " <<  sbs[i].nnz_local_c_idx[j] << " " << sbs[i].nnz_values[j] << endl;
        }
        cout << endl;
    }
    cout << endl;
}

void randomizeFloatVector(float * vec, int size){
    for(int i = 0; i < size; i++){
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        vec[i] = r;
    }
    printf("\n\n");
}

void displayFloatVector(float * vec, int size){
    for(int i = 0; i < size; i++){
        printf("%f ", vec[i] );
    }
    printf("\n\n");
}

void setZeroFloatVector(float * vec, int size){
    for(int i = 0; i < size; i++){
        vec[i] = 0.0;
    }
}

int main(){
    srand (static_cast <unsigned> (time(0)));
    ifstream datafile;
    //datafile.open("../data/blockmatrix.data");
    datafile.open("../data/data");

    int nblocks, mem_b_size, nrows, ncols;
    float density;
    datafile >> nblocks >>  nrows >> ncols >> mem_b_size;
    
    SubBlock * sbs = NULL;
    SubBlock * d_sbs;

    sbs = (SubBlock *) malloc(nblocks * sizeof(SubBlock));
    for(int i = 0; i < nblocks; i++){


        datafile >> sbs[i].nnz >> density;
        int nnz = sbs[i].nnz;

        sbs[i].nnz_global_i_idx = (int *) malloc( mem_b_size * sizeof(int));
        sbs[i].nnz_global_o_idx = (int *) malloc( mem_b_size * sizeof(int));
        
        for(int j = 0; j < mem_b_size; j++){
            datafile >> sbs[i].nnz_global_i_idx[j];
        }
        
        for(int j = 0; j < mem_b_size; j++){
            datafile >> sbs[i].nnz_global_o_idx[j];
        }


        sbs[i].nnz_local_r_idx = (int *) malloc(nnz * sizeof(int));
        sbs[i].nnz_local_c_idx = (int *) malloc(nnz * sizeof(int));
        sbs[i].nnz_values = (float *) malloc(nnz * sizeof(float));
        
        for(int j = 0; j < nnz; j++){
            datafile >> sbs[i].nnz_local_r_idx[j] >>  sbs[i].nnz_local_c_idx[j] >> sbs[i].nnz_values[j];
        }
    }
    
    #ifdef DEBUG
    printSubBlocksInfo(sbs, nblocks, mem_b_size);
    printf("The mem size of struct SubBlock is %d bytes.\n", sizeof(SubBlock));
    printf("The mem size of int is %d bytes.\n", sizeof(int));
    printf("The mem size of int * is %d bytes.\n", sizeof(int*));
    #endif

    int sbs_size = nblocks * sizeof(SubBlock);
    cudaMalloc((void **) &d_sbs, sbs_size);
    cudaMemcpy(d_sbs, sbs, sbs_size, cudaMemcpyHostToDevice);

    float * x = NULL;
    float * y = NULL;
    float * d_x = NULL;
    float * d_y = NULL;

    x = (float *) malloc(ncols * sizeof(float));
    y = (float *) malloc(nrows * sizeof(float));
    randomizeFloatVector(x, ncols * sizeof(float));
    setZeroFloatVector(y, nrows * sizeof(float));

    cudaMalloc((void **) & d_x, ncols * sizeof(float));
    cudaMalloc((void **) & d_y, nrows * sizeof(float));
    cudaMemcpy(d_x, x, ncols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, nrows * sizeof(float), cudaMemcpyHostToDevice);

    float * d_sub_y_arr = NULL;
    cudaMalloc((void **) &d_sub_y_arr, nblocks * mem_b_size * sizeof(float));
    
    CudaCompute<<<nblocks, 1>>>(d_sbs, d_x, d_y, nblocks, mem_b_size, nrows, ncols, d_sub_y_arr);
    cudaDeviceSynchronize();
    CudaMergeResults<<<1, 1>>>(d_sbs, d_x, d_y, nblocks, mem_b_size, nrows, ncols, d_sub_y_arr);

    cudaMemcpy(y, d_y, nrows * sizeof(float), cudaMemcpyDeviceToHost);

    //displayFloatVector(x, ncols);
    //displayFloatVector(y, nrows);

    printf("Hello!\n");
    return 0;
}




