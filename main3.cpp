//Combination
#include <omp.h>
#include <iostream>
#include <cmath>
#include <sys/time.h>
#include <xmmintrin.h>
#include <string>
#include <cstdlib>
#include <unistd.h>
#include <cstring>

using namespace std;

void get_A_inf(float *&A, float &A_inf, int &N)
{
    float max_line_sum = 0;
    float temp_sum = 0;
    int i, j;
#pragma omp parallel private(i, j, temp_sum) shared(A, N, max_line_sum)
    {
#pragma omp for
        for (i = 0; i < N; ++i)
        {
            temp_sum = 0;
            for (j = 0; j < N; ++j)
            {
                temp_sum += A[j * N + i];
            }
#pragma omp critical
            {
                if (temp_sum > max_line_sum)
                {
                    max_line_sum = temp_sum;
                }
            }
        }
    }
    A_inf = max_line_sum;
}

void get_A_1(float *&A, float &A_1, int &N)
{
    float max_row_sum = 0;
    float temp_sum = 0;
    int i, j;
#pragma omp parallel private(i, j, temp_sum) shared(A, max_row_sum)
    {
#pragma omp for
        for (i = 0; i < N; ++i)
        {
            temp_sum = 0;
            for (j = 0; j < N; ++j)
            {
                temp_sum += A[i * N + j];
            }
#pragma omp critical
            {
                if (temp_sum > max_row_sum)
                {
                    max_row_sum = temp_sum;
                }
            }
        }
    }
    A_1 = max_row_sum;
}

void multiply_matrices(float *B, float *A, float *BA, int N)
{

#pragma omp parallel shared(B, A, BA, N)
    {
        float *lines_sum = new float[N]();
        float *temp = new float[N]();

#pragma omp for
        for (int i = 0; i < N; i++)
        {
            memset(lines_sum, 0, N);
            for (int j = 0; j < N; j++)
            {
                __m128 B_element = _mm_set1_ps(B[i * N + j]);

                for (int k = 0; k < N; k += 4)
                {
                    __m128 A_line_part = _mm_load_ps(A + (j * N + k));
                    __m128 line_mult = _mm_mul_ps(B_element, A_line_part);
                    _mm_store_ps(temp + k, line_mult);
                }

                for (int p = 0; p < N; p += 4)
                {
                    __m128 lines_sum_part = _mm_load_ps(lines_sum + p);
                    __m128 temp_part = _mm_load_ps(temp + p);
                    __m128 lines_temp_sum = _mm_add_ps(lines_sum_part, temp_part);
                    _mm_store_ps(lines_sum + p, lines_temp_sum);
                }
            }
            memcpy(BA + i * N, lines_sum, sizeof(float) * N);
        }
        free(lines_sum);
        free(temp);
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

    int i, j, k;
#pragma omp parallel for private(i, j) shared(I, BUFF, N)
    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N; ++j)
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

#pragma omp parallel for private(i, j) shared(B, A, A_1, A_inf, N)
    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            B[i * N + j] = A[j * N + i] / (A_1 * A_inf);
        }
    }

    multiply_matrices(B, A, BA, N);

    __m128 temp;
    __m128 *I_line;
    __m128 *BA_line;
#pragma omp parallel for private(i, j, temp, I_line, BA_line) shared(R, N, BA, I)
    for (i = 0; i < N; ++i)
    {
        I_line = (__m128 *)(I + i * N);
        BA_line = (__m128 *)(BA + i * N);
        for (j = 0; j < N / 4; ++j)
        {
            temp = _mm_sub_ps(I_line[j], BA_line[j]);
            _mm_stream_ps(R + i * N + j * 4, temp);
        }
    }

    __m128 *BUFF_line;
    float *R_stack = new float[N * N]();
    float *SUMM = new float[N * N]();
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            SUMM[i * N + j] = I[i * N + j];
            R_stack[i * N + j] = I[i * N + j];
        }
    }

    for (k = 0; k < M - 1; ++k)
    {
        for (i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                TEMP[i * N + j] = 0;
            }
        }
        multiply_matrices(R_stack, R, TEMP, N);
        for (i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                R_stack[i * N + j] = TEMP[i * N + j];
            }
        }
        for (i = 0; i < N; ++i)
        {
            for (j = 0; j < N; j++)
            {
                SUMM[i * N + j] += R_stack[i * N + j];
            }
        }
    }
    multiply_matrices(SUMM, B, A_inv, N);

    delete[] B;
    delete[] I;
    delete[] R;
    delete[] BA;
    delete[] BUFF;
    delete[] TEMP;
    delete[] R_stack;
    delete[] SUMM;
    return A_inv;
}

int main(int argc, char *argv[])
{
    int world_size = omp_get_max_threads();
    cout << "World size is " << world_size << endl;
    omp_set_num_threads(world_size);
    srand(1337);

    int N = 0; //matrix size
    int M = 0; //number of iterations
    if ((argc == 1) || (argc == 2))
    {
        cout << "Please enter matrix size and number of iterations" << endl;
        return 0;
    }
    N = atoi(argv[1]);
    M = atoi(argv[2]);

    float *A = new float[N * N]();
    int i, j;
    double rand_value = 0;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            rand_value = rand() % 10;
            A[i * N + j] = rand_value;
        }
    }
    cout << "COMBINING WAY ON " << world_size << " CORES" << endl;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    float *A_inv = get_inverted_matrix(A, N, M);
    gettimeofday(&end, NULL);
    cout << "Time taken: " << ((end.tv_sec - start.tv_sec) + 0.000001 * (end.tv_usec - start.tv_usec)) << endl;

    return 0;
}
