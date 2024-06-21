#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define Numc 14
#define MAX_ITER 1e8
#define TOLERANCE 1e-6
#define dx 0.0001


// Function to perform Gauss-Seidel relaxation
void gauss_seidel(double fac, double* b, double* x, int num_iterations, int grid_size) {
    double* xn = (double*)malloc(grid_size * sizeof(double));
    for (int iter = 0; iter < num_iterations; iter++) {
        for (int j = 0; j < grid_size; j++) {
            // printf("%f  ",b[j]);
            if (j > 0 && j < grid_size - 1) {
                xn[j] = ((x[j + 1] + xn[j - 1] -  b[j]/fac) * 0.5) ;
            } else if (j == 0) {
                xn[j] = ((x[j + 1] -  b[j]/fac) * 0.5);
            } else if (j==grid_size-1) {
                xn[j] = ((xn[j - 1] -  b[j]/fac) * 0.5);
            }
             //printf("%f  ",xn[j]);// Uncomment for debugging
        }
        for (int j = 0; j < grid_size; j++) {x[j]=xn[j];}
    }
    free(xn);
}
void gauss_seidel_redblack(double fac, double* b, double* x, int num_iterations, int grid_size) {
    //double* xn = (double*)malloc(grid_size * sizeof(double));
    for (int iter = 0; iter < num_iterations; iter++) {

        for (int j = 0; j < grid_size; j+=2) {
            // printf("%f  ",b[j]);
            
            if (j > 0 && j < grid_size - 1) {
                x[j] = ((x[j + 1] + x[j - 1] -  b[j]/fac) * 0.5) ;
            } else if (j == 0) {
                x[j] = ((x[j + 1] -  b[j]/fac) * 0.5);
            } else if (j==grid_size-1) {
                x[j] = ((x[j - 1] -  b[j]/fac) * 0.5);
            }
             
        }
        for (int j = 1; j < grid_size; j+=2) {
            // printf("%f  ",b[j]);
            
            if (j > 0 && j < grid_size - 1) {
                x[j] = ((x[j + 1] + x[j - 1] -  b[j]/fac) * 0.5) ;
            } else if (j == 0) {
                x[j] = ((x[j + 1] -  b[j]/fac) * 0.5);
            } else if (j==grid_size-1) {
                x[j] = ((x[j - 1] -  b[j]/fac) * 0.5);
            }
             
        }
        //for (int j = 0; j < grid_size; j++) {x[j]=xn[j];}
    }
    //free(xn);
}


// Function to restrict the residual from fine grid to coarse grid
void restrict_residual(double* r_fine, double* r_coarse, int grid_size_fine) {
    for (int i = 0; i < grid_size_fine / 2; i++) {
    if(grid_size_fine%2==1)
        r_coarse[i] = 0.25*r_fine[2 * i] + 0.5*r_fine[2 * i + 1]+0.25*r_fine[2*i+2];
    else if(i!=grid_size_fine/2-1)
        r_coarse[i] = 0.25*r_fine[2 * i] + 0.5*r_fine[2 * i + 1]+0.25*r_fine[2*i+2];
    else
       r_coarse[i] = 0.25*r_fine[2 * i] + 0.5*r_fine[2 * i + 1]; 
    }
}

// Function to prolong the error from coarse grid to fine grid
void prolong(double* e_coarse, double* e_fine, int grid_size_fine) {
    for (int i = 0; i < grid_size_fine/2 ; i++) {
        if(i>=1)
            e_fine[2 * i] = (e_coarse[i-1]+e_coarse[i])*0.5;
        else 
            e_fine[2 * i] = e_coarse[i]*0.5;
        e_fine[2 * i + 1] = e_coarse[i];
    }
    if(grid_size_fine%2==1){
        e_fine[grid_size_fine-1]=0.5*e_coarse[grid_size_fine/2-1];
    }
}

// Function to perform a V-cycle
double v_cycle(double* f, double* u, int num_iterations, int grid_size,int numcyc) {
    // Gauss-Seidel pre-smoothing
    double fac1=pow(0.25,Numc-numcyc);
    //printf("%f ",fac);
    //for (int j = 0; j < grid_size; j++) {printf("%f ",f[j]);}
    //gauss_seidel(fac1,f,u, num_iterations, grid_size);
    if(numcyc==1){
        gauss_seidel(fac1,f,u, 4*num_iterations, grid_size);
    }
    else 
        gauss_seidel(fac1,f,u, num_iterations, grid_size);
    if(numcyc==1){
        return 0;
    }
    if(numcyc==1){
        return 0;
    }
    // Compute the residual
    double* r = (double*)malloc(grid_size * sizeof(double));
    for (int i = 0; i < grid_size; i++) {
       
        if(i>0 && i<grid_size-1){
            r[i] = f[i] - (u[i+1]-2*u[i]+u[i-1])*pow(0.25,Numc-numcyc);
        }
        else if (i==0)
           r[i] = f[i] - (u[i+1]-2*u[i])*pow(0.25,Numc-numcyc);
        else
            r[i] = f[i] - (-2*u[i]+u[i-1])*pow(0.25,Numc-numcyc);
        //printf("%f ",u[i]);
    }
    
    // Coarse grid correction

    int grid_size_coarse = grid_size / 2;
    //double** A_coarse = create_matrix(grid_size_coarse, grid_size_coarse);
    //double* f_coarse = (double*)malloc(grid_size_coarse * sizeof(double));
    //double* u_coarse = (double*)malloc(grid_size_coarse * sizeof(double));
    double* r_coarse = (double*)malloc(grid_size_coarse * sizeof(double));
    double* e_coarse = (double*)malloc(grid_size_coarse * sizeof(double));
    // Restrict the residual to the coarse grid
    restrict_residual(r, r_coarse, grid_size);
    
    // Initialize u_coarse with zeros
    for (int i = 0; i < grid_size_coarse; i++) {
        e_coarse[i] = 0.0;
    }
    /*for (int i = 0; i < grid_size_coarse; i++) {
        for (int j = 0; j < grid_size_coarse; j++) {
            if (i == j) {
                A_coarse[i][j] = -2.0*pow(0.25,Numc-numcyc+1);
            } else if (abs(i - j) == 1) {
                A_coarse[i][j] = 1.0*pow(0.25,Numc-numcyc+1);
            } else {
                A_coarse[i][j] = 0.0;
            }
        }
    }*/

    // Recursive call to v_cycle

    v_cycle( r_coarse, e_coarse, num_iterations, grid_size_coarse,numcyc-1);
    
    // Prolong the error to the fine grid
    
    
    double* e_fine = (double*)malloc(grid_size * sizeof(double));
    prolong(e_coarse, e_fine, grid_size);
    
    // Update the solution u
    for (int i = 0; i < grid_size; i++) {
        u[i] += e_fine[i];
    }
    
    //free_matrix(A_coarse, grid_size_coarse);
    //free(f_coarse);
    //free(u_coarse);
    free(r_coarse);
    free(e_coarse);
    if(numcyc!=Numc)
        free(e_fine);
    
    
    // Gauss-Seidel post-smoothing
    double fac2=pow(0.25,Numc-numcyc);
    gauss_seidel(fac2, f, u, num_iterations, grid_size);
    
    free(r);
    if (numcyc==Numc){
        double norm2=0.0;
        for (int i = 0; i < grid_size; i++) {
        
        
        norm2+=e_fine[i]*e_fine[i];
    }   
        free(e_fine);
        return sqrt(norm2);
    }
}

int main() {
    // Define the problem size and other parameters
    clock_t start_time, end_time;
    double cpu_time_used;
    start_time = clock();
    int numcyc=Numc;
    int grid_size = 2 / dx + 1 - 2;
    int num_iterations = 3;
    double norm2=1+TOLERANCE;
    // Create tridiagonal matrix A and source term f
    //double** A = create_matrix(grid_size, grid_size);
    double* f = (double*)malloc(grid_size * sizeof(double));
    // Fill A and f with appropriate values
    /*for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            if (i == j) {
                A[i][j] = -2.0;
            } else if (abs(i - j) == 1) {
                A[i][j] = 1.0;
            } else {
                A[i][j] = 0.0;
            }
        }
    }*/
    
    // Fill f (example)
    for (int i = 0; i < grid_size; i++) {
        if (i==0||i==grid_size-1){
            f[i]=1.0*dx*dx-1.0;
            }
        else
        f[i] = 1.0*dx*dx; // or any desired values
    }
    
    // Initialize solution vector u with zeros
    double* u = (double*)malloc(grid_size * sizeof(double));
    for (int i = 0; i < grid_size; i++) {
        u[i] = 0.0;
    }
    //for (int i = 0; i < grid_size; i++) {
    // Perform V-cycle multigrid method
    int iter=0;
    while(norm2>TOLERANCE && iter<MAX_ITER){
        norm2=v_cycle(f, u, num_iterations, grid_size,numcyc);
        iter+=1;
    }
    // Print or use the resulting solution u
    for (int i = 0; i < grid_size; i++) {
        printf("%f \n", u[i]);
    }
    printf("\n converged in %d",iter-1);
    // Free allocated memory
    //free_matrix(A, grid_size);
    free(f);
    free(u);
    end_time = clock();

    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    printf("\nCPU time used: %f seconds\n", cpu_time_used);

    
    return 0;
}
