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


if(proc_rank == 0){
double* A_x = new double[N * N]();
generate_matrix(A);

generate_vector(x);

mul_matr_vec(A, x, A_x);

sum_vector(b, A_x, -1, r);

for(int i = 0; i < N; i++) {
z[i] = r[i];
}

delete[] A_x;
}
MPI_Bcast(r, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

//START OF MATRIX SCATTERING PREPARATIONS---------------------------------------------------------------------------------------------
int part_size = N / world_size;
double* A_part = new double[part_size * N]();
double* A_z = new double[N]();
MPI_Bcast(z, N,MPI_DOUBLE, 0, MPI_COMM_WORLD);

//END OF MATRIX SCATTERING PREPARATIONS-----------------------------------------------------------------------------------------------
MPI_Scatter(A, part_size *N, MPI_DOUBLE, A_part, part_size * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
delete[] A;

double alpha = 0;
double beta = 0;
double* alpha_z = new double[N]();
double* beta_z = new double[N]();

double* r_prev = new double[N]();
double* temp_result = new double[part_size]();

while((vector_norm(r) / vector_norm(b)) > EPS) {
for(int i = 0; i < part_size; i++) {
temp_result[i] = 0;
for(int j = 0; j < N; j++) {
temp_result[i] += A_part[j + N * i] * z[j];
}
}
MPI_Allgather(temp_result,part_size, MPI_DOUBLE, A_z, part_size, MPI_DOUBLE, MPI_COMM_WORLD);

alpha = dot_product(r, r) / dot_product(A_z, z);

mul_vect_scal(z, alpha, alpha_z);
sum_vector(x, alpha_z, 1, x);
memcpy(r_prev, r, sizeof(double) * N);

mul_vect_scal(A_z, alpha, A_z);

sum_vector(r, A_z, -1, r);


beta = dot_product(r,r) / dot_product(r_prev, r_prev);

mul_vect_scal(z, beta, beta_z);
sum_vector(r, beta_z, 1, z);

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
if(world_rank == 0){
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
cout <<"Time taken: " << end - start << endl;
}
MPI_Finalize();
return 0;
}