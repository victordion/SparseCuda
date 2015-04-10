#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
using namespace std;

struct SubBlock{
    
    int * nnz_global_i_idx;
    int * nnz_global_o_idx;

    int nnz;
    int * nnz_local_r_idx;
    int * nnz_local_c_idx;
    float * nnz_values;
};

__global__ void CudaCompute(SubBlock * sbs, float * x, float * y){
        SubBlock * sb = &sbs[blockIdx.x];
        printf("Hi!\n");
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

int main(){
    ifstream datafile;
    //datafile.open("../data/blockmatrix.data");
    datafile.open("../data/data");

    int nblocks, mem_b_size, nrows, ncols;
    float density;
    datafile >> nblocks >>  nrows >> ncols >> mem_b_size;
    
    SubBlock * sbs = NULL;

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
    
    
    printSubBlocksInfo(sbs, nblocks, mem_b_size);

    float * y = NULL;
    float * x = NULL;
    CudaCompute<<<10,1>>>(sbs,x,y);
    printf("Hello!\n");
    return 0;
}




