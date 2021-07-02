#include <mpi.h>
#include <iostream>
#include <string>
#include <cstring>
#include <cmath>
#include <ctime>
#include <unistd.h>//for sleep()

using namespace std;

int N = 6400;
#define EPS 0.0000001


double dot_product(double* a, double* b) {
    double product = 0;
    for(unsigned int i = 0; i < N; i++) {
        product += a[i] * b[i];
    }
    return product;
}

void mul_vect_scal(double* a, double scal, double* result) {
    for(int i = 0; i < N; i++) {
        result[i] = scal * a[i];
    }
}

void sum_vector(double* a, double* b, int sign, double* result) {
    if(sign == 1) {
        for(int i = 0; i < N; i++) {
            result[i] = a[i] + b[i];
        }
    }else if (sign == -1) {
        for(int i = 0; i < N; i++) {
            result[i] = a[i] - b[i];
        }
    }
}

void mul_matr_vec(double* A, double* b, double* result) {
    for(unsigned int i = 0; i < N; i++) {
        result[i] = 0;
        for(unsigned int j = 0; j < N; j++) {
            result[i] += A[i * N + j] * b[j];
        }
    }
}

void generate_matrix(double* matrix) {
    double rand_value = 0;
    for(unsigned int i = 0; i < N; i++) {
        for(unsigned int j = 0; j < N; j++) {
            if(i > j) {
                continue;
            }
            rand_value = rand() % 10;
            matrix[i * N + j] = rand_value;
            if(i != j) {
                matrix[j * N + i] = rand_value;
            }else{
                matrix[i * N + j] += 470;
            }
        }
    }
}

void generate_vector(double* vector) {
    double rand_value = 0;
    for(unsigned int i = 0; i < N; i++) {
        rand_value = rand() % 10;
        vector[i] = rand_value;
    }
}

double vector_norm(double* a) {
    double result = 0;
    for(int i = 0; i < N; i++) {
        result += a[i] * a[i];
    }
    return sqrt(result);
}

void gradient_solution(int proc_rank, int world_size) {
    double* x = new double[N]();
    double* z = new double[N]();
    double* r = new double[N]();
    double* A = new double[N * N]();
    double* b = new double[N]();
    generate_vector(b);

    if(proc_rank == 0) {
        // double* A_x = new double[N * N]();//result

        generate_matrix(A);

        generate_vector(x);

    }
    //arrays for sending
    int number_of_columns_on_processor = N / world_size;
    int* scatter_indexes_vec = new int[world_size];
    int* scatter_indexes_mat = new int[world_size];
    int* send_counts_vec = new int[world_size];
    int* send_counts_mat = new int[world_size];
    for(int i = 0; i < world_size; i++) {
        scatter_indexes_vec[i] = i * number_of_columns_on_processor;
        scatter_indexes_mat[i] = i * number_of_columns_on_processor * N;
        send_counts_vec[i] = number_of_columns_on_processor;
        send_counts_mat[i] = number_of_columns_on_processor * N;
    }
    //--------------------

    //arrays for recieving
    double* b_part = new double[number_of_columns_on_processor];
    double* x_part = new double[number_of_columns_on_processor];
    double* A_part = new double[number_of_columns_on_processor * N]();
    //--------------------

    //arrays for calculating
    double* proc_mult_result = new double[N]();
    double* reducing_result = new double[N]();
    double* reduce_part = new double[number_of_columns_on_processor]();
    //--------------------

    //scattering
    MPI_Scatterv(b, send_counts_vec, scatter_indexes_vec, MPI_DOUBLE, b_part, number_of_columns_on_processor, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(x, send_counts_vec, scatter_indexes_vec, MPI_DOUBLE, x_part, number_of_columns_on_processor, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(A, send_counts_mat, scatter_indexes_mat, MPI_DOUBLE, A_part, number_of_columns_on_processor * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //--------------------

    //calculating
    for(int np = 0; np < number_of_columns_on_processor; np++) {
        for(int i = 0; i < N; i++) {
            proc_mult_result[i] += x_part[np] * A_part[i + N * np];
        }
    }
    MPI_Reduce(proc_mult_result, reducing_result, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Scatterv(reducing_result, send_counts_vec, scatter_indexes_vec, MPI_DOUBLE, reduce_part, number_of_columns_on_processor, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for(int i = 0; i < number_of_columns_on_processor; i++) {
        b_part[i] -= reduce_part[i];
    }
    MPI_Allgather(b_part, number_of_columns_on_processor, MPI_DOUBLE, r, number_of_columns_on_processor, MPI_DOUBLE, MPI_COMM_WORLD);
    //--------------------

    //deleting unnecesary arrays
    delete[] b_part;
    delete[] x_part;
    delete[] proc_mult_result;
    delete[] reducing_result;
    delete[] reduce_part;
    //--------------------
    MPI_Bcast(r, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for(int i = 0; i < N; i++) {
        z[i] = r[i];
    }

    //START OF MATRIX SCATTERING PREPARATIONS---------------------------------------------------------------------------------------------
    double* A_z = new double[N]();
    MPI_Bcast(z, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //END OF MATRIX SCATTERING PREPARATIONS-----------------------------------------------------------------------------------------------
    MPI_Scatter(A, number_of_columns_on_processor * N, MPI_DOUBLE, A_part, number_of_columns_on_processor * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    delete[] A;

    double alpha = 0;
    double beta = 0;
    double* alpha_z = new double[N]();
    double* beta_z = new double[N]();
    double* r_prev = new double[N]();
    double* temp_result = new double[number_of_columns_on_processor]();

    while((vector_norm(r) / vector_norm(b)) > EPS) {
        //START Paral. mult.
        for(int i = 0; i < number_of_columns_on_processor; i++) {
            temp_result[i] = 0;
            for(int j = 0; j < N; j++) {
                temp_result[i] += A_part[j + N * i] * z[j];
            }
        }
        MPI_Allgather(temp_result, number_of_columns_on_processor, MPI_DOUBLE, A_z, number_of_columns_on_processor, MPI_DOUBLE, MPI_COMM_WORLD);
        //END Paral. mult.
        alpha = dot_product(r, r) / dot_product(A_z, z);

        mul_vect_scal(z, alpha, alpha_z);
        //START x calculating-------------------------
        sum_vector(x, alpha_z, 1, x);
        //END x calculating---------------------------
        memcpy(r_prev, r, sizeof(double) * N);

        mul_vect_scal(A_z, alpha, A_z);

        sum_vector(r, A_z, -1, r);


        beta = dot_product(r,r) / dot_product(r_prev, r_prev);

        mul_vect_scal(z, beta, beta_z);
        sum_vector(r, beta_z, 1, z);

    }
    if(proc_rank == 0) {
        double* A_result = new double[N]();
        cout << "Proofing result" << endl;
        mul_matr_vec(A, x, A_result);
        cout << "A * x is " << endl;
        for(int i = 0; i < N; i++) {
            cout << A_result[i] << " ";
        }
        cout << endl;
        cout << "b is " << endl;
        for(int i = 0; i < N; i++) {
            cout << b[i] << " ";
        }
        cout << endl;
    }
    delete[] x;
    delete[] r;
    delete[] z;
    delete[] b;
    delete[] A_z;
    delete[] A_part;
    delete[] alpha_z;
    delete[] beta_z;
    delete[] r_prev;
    delete[] temp_result;
}

int main(int argc, char** argv) {
    MPI_Init( &argc , &argv);

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int name_len = 0;
    MPI_Get_processor_name(proc_name, &name_len);
    if(N % world_size != 0 ) {
        if(world_rank == 0) {
            cout << "Please, restart the programm with the size of matrix, that can be divided by the world size" << endl;
        }
        return 0;
    }
    srand(1337);
    double start, end;
    start = MPI_Wtime();
    gradient_solution(world_rank, world_size);
    end = MPI_Wtime();
    if(world_rank == 0) {
        cout << "Time taken: " << end - start << endl;
    }
    MPI_Finalize();
    return 0;
}
