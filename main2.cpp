//Vectorization
#include <iostream>
#include <cmath>
#include <sys/time.h>
#include <xmmintrin.h>
#include <string>
#include <cstdlib>

using namespace std;

void get_A_inf(float *&A, float &A_inf, int &N)
{
    float max_line_sum = 0;
    float temp_sum = 0;
    for (int i = 0; i < N; ++i)
    {
        temp_sum = 0;
        for (int j = 0; j < N; ++j)
        {
            temp_sum += A[j * N + i];
        }
        if (temp_sum > max_line_sum)
        {
            max_line_sum = temp_sum;
        }
    }
    A_inf = max_line_sum;
}

void get_A_1(float *&A, float &A_1, int &N)
{
    float max_row_sum = 0;
    float temp_sum = 0;
    for (int i = 0; i < N; ++i)
    {
        temp_sum = 0;
        for (int j = 0; j < N; ++j)
        {
            temp_sum += A[i * N + j];
        }
        if (temp_sum > max_row_sum)
        {
            max_row_sum = temp_sum;
        }
    }
    A_1 = max_row_sum;
}

void multiply_matrices(float *&A, float *&B, float *&AB, int &N)
{
    __m128 temp_sum;
    __m128 temp_mul;
    float line_sum;
    float summator[4];
    float *B_trans = new float[N * N];
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            B_trans[i * N + j] = B[j * N + i];
        }
    }
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; j ++)
        {
            __m128 *A_p = (__m128 *) &A[i * N];
            __m128 *B_tran_p = (__m128 *) &B_trans[j * N];
            temp_sum = _mm_setzero_ps();
            for(int k = 0; k < N / 4; k++)
            {
                temp_mul = _mm_mul_ps(A_p[k], B_tran_p[k]);
                temp_sum = _mm_add_ps(temp_sum, temp_mul);
            }
            _mm_store_ps(summator, temp_sum);
            line_sum = 0;
            for(int p = 0; p < 4; p++)
            {
                line_sum += summator[p];
            }
            AB[i * N + j] = line_sum;

        }
    }
}

float *get_inverted_matrix(float *&A, int N, int M)
{
    float *B = new float[N * N]();
    float A_inf = 0;
    float A_1 = 0;
    float *A_inv = new float[N * N]();
    float *I = new float[N * N]();
    float *R = new float[N * N]();
    float *BA = new float[N * N]();
    float *BUFF = new float[N * N]();
    float *TEMP = new float[N * N]();

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (i == j)
            {
                I[i * N + j] = 1;
                BUFF[i * N + j] = 1;
            }
        }
    }

    get_A_1(A, A_1, N);
    get_A_inf(A, A_inf, N);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            B[i * N + j] = A[j * N + i] / (A_1 * A_inf);
        }
    }

    multiply_matrices(A, B, BA, N);

    __m128 temp;
    __m128 *I_line;
    __m128 *BA_line;
    for (int i = 0; i < N; ++i)
    {
        I_line = (__m128 *)(I + i * N);
        BA_line = (__m128 *)(BA + i * N);
        for (int j = 0; j < N / 4; ++j)
        {
            temp = _mm_sub_ps(I_line[j], BA_line[j]);
            _mm_stream_ps(R + i * N + j * 4, temp);
        }
    }

    for (int k = 0; k < M - 1; ++k)
    {
        std::cout << "Iteration " << k << "." << std::endl;
        multiply_matrices(I, R, TEMP, N);
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; j++)
            {
                I[i * N + j] = TEMP[i * N + j];
            }
        }
        __m128 temp;
        __m128 *BUFF_line;
        __m128 *I_line;
        for (int i = 0; i < N; ++i)
        {
            I_line = (__m128 *)(I + i * N);
            BUFF_line = (__m128 *)(BUFF + i * N);
            for (int j = 0; j < N / 4; ++j)
            {
                temp = _mm_add_ps(I_line[j], BUFF_line[j]);
                _mm_stream_ps(BUFF + i * N + j * 4, temp);
            }
        }
    }
    multiply_matrices(BUFF, B, A_inv, N);

    free(B);
    free(I);
    free(R);
    free(BA);
    free(BUFF);
    free(TEMP);
    return A_inv;
}

int main(int argc, char *argv[]) {
    int N = 0; //matrix size
    int M = 0; //number of iterations
    if ((argc == 1) || (argc == 2)) {
        std::cout << "Please enter matrix size and number of iterations" << std::endl;
        return 0;
    }
    N = std::atoi(argv[1]);
    M = std::atoi(argv[2]);

    float *A = new float[N * N]();
    int i, j;
    double rand_value = 0;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i > j) {
                continue;
            }
            rand_value = rand() % 10;
            A[i * N + j] = rand_value;
            if (i != j) {
                A[j * N + i] = rand_value;
            } else {
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