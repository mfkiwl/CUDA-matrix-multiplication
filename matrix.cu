#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLK 32
#define dimA (10*BLK*10*BLK)
#define dimB (10*BLK*20*BLK)
#define szA (10*BLK*10*BLK*sizeof(double))
#define szB (10*BLK*20*BLK*sizeof(double))
#define A ((const double (*)[10*BLK])a)
#define B ((const double (*)[20*BLK])b)
#define C ((double (*)[20*BLK])c)
#define bx blockIdx.x
#define by blockIdx.y
#define tx threadIdx.x
#define ty threadIdx.y

void init(int n,double *M){
    int i;
    for(i=0;i<n;i++){
        M[i]=(double)rand()/RAND_MAX;
    }
}

void host_mm(const double *a,const double *b,double *c){
    int i,j,k;
    for(i=0;i<10*BLK;i++){
        for(j=0;j<20*BLK;j++){
            for(k=0;k<10*BLK;k++){
                C[i][j]+=A[i][k]*B[k][j];
            }
        }
    }
}

void print(double *c){
    int i,j;
    for(i=0;i<10*BLK;i++){
        for(j=0;j<20*BLK;j++){
            printf("%.2f\t",C[i][j]);
        }
        printf("\n");
    }
}

__global__
void device_mm(const double *a,const double *b,double *c){
    int k;
    for(k=0;k<10*BLK;k++)
        C[bx*BLK+tx][by*BLK+ty]+=A[bx*BLK+tx][k]*B[k][by*BLK+ty];
}

__global__
void tiled_device_mm(const double *a,const double *b,double *c){
    __shared__ double sA[BLK][BLK];
    __shared__ double sB[BLK][BLK];
    int si,i;
    double sum=0;
    for(si=0;si<10;si++){
        sA[tx][ty]=A[bx*BLK+tx][si*BLK+ty];
        sB[tx][ty]=B[si*BLK+tx][by*BLK+ty];
        __syncthreads();
        for(i=0;i<BLK;i++){
            sum+=sA[tx][i]*sB[i][ty];
        }
        __syncthreads();
    }
    C[bx*BLK+tx][by*BLK+ty]=sum;
}

void check(int n,double *x,double *y){
    int i;
    double maxerr=0;
    for(i=0;i<n;i++){
        if(fabsf(x[i]-y[i])/y[i]>maxerr){
            maxerr=fabsf(x[i]-y[i])/y[i];
        }
    }
    printf("max err = %g\n",maxerr);
}

int main(){
    clock_t start,finish;
    double hosttime,devicetime;

    dim3 th(BLK,BLK);
    dim3 bl(10,20);

    double *hA,*hB,*rC,*dA,*dB,*dC,*hC;
    hA=(double*)malloc(szA);
    hB=(double*)malloc(szB);
    hC=(double*)malloc(szB);
    rC=(double*)malloc(szB);

    init(dimA,hA);
    init(dimB,hB);
    memset(hC,0,szB);

    start=clock();
    host_mm(hA,hB,hC);
    finish=clock();
    hosttime=(double)(finish-start)/CLOCKS_PER_SEC;
    printf("host: %.3f\n",hosttime);

    cudaMalloc(&dA,szA);
    cudaMalloc(&dB,szB);
    cudaMalloc(&dC,szB);
    cudaMemcpy(dA,hA,szA,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,szB,cudaMemcpyHostToDevice);
    cudaMemset(dC,0,szB);
    start=clock();
    device_mm<<<bl,th>>>(dA,dB,dC);
    cudaThreadSynchronize();
    finish=clock();
    cudaMemcpy(rC,dC,szB,cudaMemcpyDeviceToHost);
    devicetime=(double)(finish-start)/CLOCKS_PER_SEC;
    printf("device: %.3f, speedup=%.3f\n",devicetime,hosttime/devicetime);
    check(dimB,rC,hC);
    //print(rC);
    //printf("CPU:\n");
    //print(hC);

    cudaMemcpy(dA,hA,szA,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,szB,cudaMemcpyHostToDevice);
    cudaMemset(dC,0,szB);
    start=clock();
    tiled_device_mm<<<bl,th>>>(dA,dB,dC);
    cudaThreadSynchronize();
    finish=clock();
    cudaMemcpy(rC,dC,szB,cudaMemcpyDeviceToHost);
    devicetime=(double)(finish-start)/CLOCKS_PER_SEC;
    printf("tiled_device: %.3f, speedup=%.3f\n",devicetime,hosttime/devicetime);
    check(dimB,rC,hC);

    free(hA);
    free(hB);
    free(hC);
    free(rC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}