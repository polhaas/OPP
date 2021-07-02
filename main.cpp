#include <iostream>
#include <cmath>
#include <sys/time.h>
#include <cstdlib>

using namespace std;

void get_A_inf(float*& A, float& A_inf, int& N){
    float max_line_sum = 0;
    float temp_sum = 0;
    for(int i = 0; i < N; ++i){
        temp_sum = 0;
        for(int j = 0; j < N; ++j){
            temp_sum += A[j * N + i];
        }
        if(temp_sum > max_line_sum){
            max_line_sum = temp_sum;
        }
    }
    A_inf = max_line_sum;
}

void get_A_1(float*& A, float& A_1, int& N){
    float max_row_sum = 0;
    float temp_sum = 0;
    for(int i = 0; i < N; ++i){
        temp_sum = 0;
        for(int j = 0; j < N; ++j){
            temp_sum += A[i * N + j];
        }
        if(temp_sum > max_row_sum){
            max_row_sum = temp_sum;
        }
    }
    A_1 = max_row_sum;
}

void multiply_matrices(float*& A, float*& B, float*& AB, int& N){
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            for(int k = 0; k < N; ++k){
                AB[i * N + k] += A[i * N + j] * B[j * N + k];
            }
        }
    }
}

float* get_inverted_matrix(float*& A, int N, int M){
    float* B = new float [N*N]();
    float A_inf = 0;
    float A_1 = 0;
    float* A_T = new float [N*N]();
    float* A_inv = new float [N*N]();
    float* I = new float [N*N]();
    float* R = new float [N*N]();
    float* BA = new float [N*N]();
    float* BUFF = new float [N*N]();

    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            if(i == j){
                I[i * N + j] = 1;
                BUFF[i * N + j] = 1;
                A_T[i * N + j] = A[i * N + j];
            }
            if(i < j){
                A_T[i * N + j] = A[j * N + i];
                A_T[i * N + j] = A[i * N + j];
            }
        }
    }

    get_A_1(A, A_1, N);
    get_A_inf(A, A_inf, N);

    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            B[i * N + j] = A[j * N + i] / (A_1 * A_inf);
        }
    }

    multiply_matrices(A, B, BA, N);

    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            float temp = I[i * N + j] - BA[i * N + j];
            R[i * N + j] = temp;
        }
    }

    float* TEMP = new float [N*N]();

    for(int k = 0; k < M-1; ++k){
        std::cout << "Iteration " << k << "." << std::endl;
        multiply_matrices(I, R, TEMP, N);
        for(int i = 0; i < N; ++i){
            for(int j = 0; j < N; j++){
                I[i * N + j] = TEMP[i * N + j];
                TEMP[i * N + j] = 0;
                BUFF[i * N + j] += I[i * N + j];
            }
        }
    }
    multiply_matrices(BUFF, B, A_inv, N);
    return A_inv;
}

int main(int argc, char* argv[]){
    srand(1337);

    int N = 0; //matrix size
    int M = 0; //number of iterations
    if ((argc == 1) || (argc == 2))
    {
        cout << "Please enter matrix size and number of iterations" << endl;
        return 0;
    }
    N = stoi(argv[1]);
    M = stoi(argv[2]);

    float *A = new float[N * N]();
    int i, j;
    double rand_value = 0;
    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
            if(i > j) {
                continue;
            }
            rand_value = rand() % 10;
            A[i * N + j] = rand_value;
            if(i != j) {
                A[j * N + i] = rand_value;
            }else{
                A[i * N + j] += 470;
            }
        }
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);
    get_inverted_matrix(A, N, M);
    gettimeofday(&end, NULL);
    cout << "Time taken: " << ((end.tv_sec - start.tv_sec) + 0.000001 * (end.tv_usec - start.tv_usec)) << endl;
    return 0;
}
