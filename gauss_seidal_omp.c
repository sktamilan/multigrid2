#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_ITER 1e8
#define TOLERANCE 1e-6
#define dx 0.001
#include <omp.h>
#define p 8

double gauss_seidelr(double fac, double* b, double* x, int grid_size) {
    double* xn = (double*)malloc(grid_size * sizeof(double));
    int iter=0;
    double norm2=1.0+ TOLERANCE;

    while(iter<MAX_ITER && norm2>TOLERANCE) {
        #pragma omp parallel for num_threads(p)
        for (int j = 0; j < grid_size; j+=2) {
            // printf("%f  ",b[j]);
            
            if (j > 0 && j < grid_size - 1) {
                xn[j] = ((x[j + 1] + x[j - 1] -  b[j]/fac) * 0.5) ;
            } else if (j == 0) {
                xn[j] = ((x[j + 1] -  b[j]/fac) * 0.5);
            } else if (j==grid_size-1) {
                xn[j] = ((x[j - 1] -  b[j]/fac) * 0.5);
            }
             
        }
        #pragma omp parallel for num_threads(p)
        for (int j = 1; j < grid_size; j+=2) {
            // printf("%f  ",b[j]);
            
            if (j > 0 && j < grid_size - 1) {
                xn[j] = ((xn[j + 1] + xn[j - 1] -  b[j]/fac) * 0.5) ;
            } else if (j == 0) {
                xn[j] = ((xn[j + 1] -  b[j]/fac) * 0.5);
            } else if (j==grid_size-1) {
                xn[j] = ((xn[j - 1] -  b[j]/fac) * 0.5);
            }
        }
        
    
        norm2=0.0;

        #pragma omp parallel for num_threads(p) reduction(+:norm2)
        for (int i = 0; i < grid_size; i++) {
        
        
        norm2+=(xn[i]-x[i])*(xn[i]-x[i]);}
        norm2=sqrt(norm2);
     
    iter+=1;
    #pragma omp parallel for num_threads(p)
    for (int j = 0; j < grid_size; j++) {x[j]=xn[j];}
    }
    free(xn);
    printf("converged in %d \n",iter-1);
    return  norm2;
}

int main() {
    
    // Define the problem size and other parameters
    double start,end;
    start=omp_get_wtime();
    
    int grid_size = 2 / dx + 1 - 2;
   
    double norm2=1+TOLERANCE;
    // Create tridiagonal matrix A and source term f
    //double** A = create_matrix(grid_size, grid_size);
    double* f = (double*)malloc(grid_size * sizeof(double));
  
    // Fill f (example)
    #pragma omp parallel for num_threads(p)
    for (int i = 0; i < grid_size; i++) {
        if (i==0||i==grid_size-1){
            f[i]=1.0*dx*dx-1.0;
            }
        else
        f[i] = 1.0*dx*dx; // or any desired values
    }
    
    // Initialize solution vector u with zeros
    double* u = (double*)malloc(grid_size * sizeof(double));
    #pragma omp parallel for num_threads(p)
    for (int i = 0; i < grid_size; i++) {
        u[i] = 0.0;
    }
    //for (int i = 0; i < grid_size; i++) {
    // Perform V-cycle multigrid method
   
    norm2=gauss_seidelr(1,f, u, grid_size);
    
    end=omp_get_wtime();
    
    // Print or use the resulting solution u
    // for (int i = 0; i < grid_size; i++) {
    //     printf("%f \n", u[i]);
    // }
    //printf("\n converged in %d",iter-1);
    
    free(f);
    free(u);
    
    printf("\nWall time used: %f seconds\n", end-start);

    
    return 0;
}
