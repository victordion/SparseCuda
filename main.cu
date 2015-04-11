#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include <ctime>
#include <cuda.h>
//#define DEBUG
//#define HANDLE_ERROR(x) if((x) != 0) cout << "Error!" << endl;

using namespace std;

struct SubBlock{
    
    int * nnz_global_i_idx;
    int * nnz_global_o_idx;

    int nnz;
    int * nnz_local_r_idx;
    int * nnz_local_c_idx;
    float * nnz_values;
};
//void printSubBlocksInfo(SubBlock * sbs, int nsbs, int mem_b_size);

__global__ void CudaCompute(SubBlock * d_sbs, float * d_x, float * d_y, int nblocks, int mem_b_size, int nrows, int ncols , float * sub_y_arr){
        /*
            sub_y_arr stores float number, with nblocks rows, mem_b_size columns
        */
        //#ifdef DEBUG
        //printf("This is Cuda Block # %d: \n", blockIdx.x);
        //#endif

        //if(blockIdx.x >= nblocks)
        //    return;

        
        //SubBlock * work_sb = &d_sbs[blockIdx.x];


        //printSubBlocksInfo(work_sb, 1, mem_b_size);

        /*
        float * x_sub = (float *) malloc(mem_b_size * sizeof(float));
        float * y_sub = (float *) malloc(mem_b_size * sizeof(float));
        //float * x;


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
            int x_sub_idx = work_sb->nnz_local_c_idx[i] - 1;
            int y_sub_idx = work_sb->nnz_local_r_idx[i] - 1;
            y_sub[y_sub_idx] += work_sb->nnz_values[i] * x_sub[x_sub_idx];
            //#ifdef DEBUG
            //    printf("This is Cuda Block # %d:  Computing (%d, %d) product as (%f)\n", blockIdx.x, x_sub_idx, y_sub_idx, work_sb->nnz_values[i] * x_sub[x_sub_idx]);
            //#endif
        }

        for(int i = 0; i < mem_b_size; i++){
            sub_y_arr[blockIdx.x * mem_b_size + i] = y_sub[i];
        }
        */

}

__global__ void CudaMergeResults(SubBlock * d_sbs, float * d_x, float * d_y, int nblocks, int mem_b_size, int nrows, int ncols , float * sub_y_arr){
    if(blockIdx.x == 0 && threadIdx.x == 0){
        for(int i = 0; i < nblocks; i++){
            int * outLocs = d_sbs[i].nnz_global_o_idx;
            for(int j = 0; j < mem_b_size; j++){
            
                d_y[outLocs[j] - 1] += sub_y_arr[i * mem_b_size + j];
            }
        }
    }
}

__global__ void cudaDummy(){
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

__host__ void randomizeFloatVector(float * vec, int size){
    for(int i = 0; i < size; i++){
        float r = ((float) rand()) /  (RAND_MAX);
        *(vec+i) = r;
    }
    //printf("\n\n");
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

    int count;
    cudaGetDeviceCount(&count);
    cout << "There are " << count << " GPU devices available. " << endl;  
    cudaSetDevice(1);  

    srand ( time(0));
    ifstream datafile;
    //datafile.open("../data/blockmatrix.data");
    datafile.open("../data/data");

    int nblocks, mem_b_size, nrows, ncols;
    float density;
    datafile >> nblocks >>  nrows >> ncols >> mem_b_size;
    

    float * x;
    float * y;
    float * d_x;
    float * d_y;

    x = (float *)  malloc(ncols * sizeof(float));
    y = (float *) malloc(nrows * sizeof(float));
    
    
    // fixed a nightmare bug here on 04/11/2015
    // originally it was:
    // randomizeFloatVector(x, ncols * sizeof(float) );
    randomizeFloatVector(x, ncols );
    setZeroFloatVector(y, nrows );

    
    int d_x_size = ncols * sizeof(float);
    int d_y_size = nrows * sizeof(float);
    
    cudaMalloc((void **) &d_x, d_x_size) ;
    int ErrorCode = cudaGetLastError();
    cout << ErrorCode << endl;

    

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





    cudaMalloc((void **) &d_y, d_y_size);
    cudaMemcpy(d_x, x, ncols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, nrows * sizeof(float), cudaMemcpyHostToDevice);

    float * d_sub_y_arr = NULL;
    cudaMalloc((void **) &d_sub_y_arr, nblocks * mem_b_size * sizeof(float));
    

    CudaCompute<<<30, 1>>>(d_sbs, d_x, d_y, nblocks, mem_b_size, nrows, ncols, d_sub_y_arr);
    ErrorCode = cudaGetLastError();
    cout << "CudaCompute Error Code is " << endl; 
    cout << ErrorCode << endl;
    
    cudaDummy<<<1, 1>>>();
    ErrorCode = cudaGetLastError();
    cout << "CudaCompute Error Code is " << endl; 
    cout << ErrorCode << endl;

    //cudaDeviceSynchronize();
    CudaMergeResults<<<1, 1>>>(d_sbs, d_x, d_y, nblocks, mem_b_size, nrows, ncols, d_sub_y_arr);
    ErrorCode = cudaGetLastError();
    cout << "CudaCompute Error Code is " << endl; 
    cout << ErrorCode << endl;

    cudaMemcpy(y, d_y, nrows * sizeof(float), cudaMemcpyDeviceToHost);

    //displayFloatVector(x, ncols);
    //displayFloatVector(y, nrows);
        
    free(x);
    free(y);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_sub_y_arr);

    cudaFree(d_sbs);
    datafile.close();
    printf("Hello!\n");
    return 0;
}




