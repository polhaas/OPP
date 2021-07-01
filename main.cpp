#include <mpi.h>
#include <iostream>
#include <time.h>
#include <unistd.h>

using namespace std;

void generate_matrix(double* matrix, int num_rows, int num_cols) {
    double rand_value = 0;
    for(unsigned int i = 0; i < num_rows; i++) {
        for(unsigned int j = 0; j < num_cols; j++) {
            rand_value = rand() % 10;
            matrix[i * num_cols + j] = rand_value;
        }
    }
}

int main(int argc, char** argv) {
    srand(3228);
    MPI_Init(&argc, &argv);

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


    int ndims = 2;
    double start, end;
    start = MPI_Wtime();
    //создание решётки
    int dims[ndims];
    dims[0] = 1;
    dims[1] = 16;

    int periods[2];
    periods[0] = 0;
    periods[0] = 0;

    int reorder = 0;

    MPI_Comm table_comm;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &table_comm);
    //----------------
    int n1 = 4096;
    int n2 = 2048;
    int n3 = 4096;
    double* A;
    double* B;
    double* AB;
    if(world_rank == 0) {
        A = (double*)malloc(n1 * n2 * sizeof(double));
        B = (double*)malloc(n2 * n3 * sizeof(double));
        AB = (double*)malloc(n1 * n3 * sizeof(double));
        generate_matrix(A, n1, n2);
        generate_matrix(B, n2, n3);
        for(int i = 0; i < n1; i++) {
            for(int j = 0; j < n3; j++) {
                AB[i * n3 + j] = 0;
            }
        }
    }
    //коммуникатор для строк
    int coords[2];
    int proc_table_rank = 0;
    MPI_Comm_rank(table_comm, &proc_table_rank);
    MPI_Cart_coords(table_comm, proc_table_rank, 2, coords);
    /*
     * coords[0] - rows
     * coords[1] - collumns
     */
    MPI_Comm row_comm;
    MPI_Comm_split(table_comm, coords[1], coords[0], &row_comm);
    //----------------------

    //коммуникатор для столбцов
    MPI_Comm col_comm;
    MPI_Comm_split(table_comm, coords[0], coords[1], &col_comm);
    //-------------------------
    //разрезание матриц
    /**
     * dim[0] - число строк
     * dim[1] - число столбцов
     */
    double A_rows[n1 * n2 / dims[1]];
    if(coords[0] == 0) { //нулевой столбец
        MPI_Scatter( A , n1 * n2 / dims[1] , MPI_DOUBLE , A_rows , n1 * n2 / dims[1] , MPI_DOUBLE , 0 , col_comm);
    }
    double B_cols[n2 * n3 / dims[0]];
    if(coords[1] == 0) { //нулевая строка
        MPI_Datatype b_part_vec;
        //^изначальный тип для разрезания матрицы В, но проблема в том, что Scatter не будет правильно воспринимать размеры этого типа
        MPI_Type_vector(n2, n3 / dims[0], n3, MPI_DOUBLE, &b_part_vec);
        MPI_Type_commit(&b_part_vec);

        MPI_Datatype b_part_for_tricking_scatter;
        MPI_Type_create_resized(b_part_vec, 0, sizeof(double) * n3 / dims[0], &b_part_for_tricking_scatter);
        //^меняем размер типа данных, чтобы таким образом "обмануть" Scatter
        MPI_Type_commit(&b_part_for_tricking_scatter);
        MPI_Scatter(B, 1, b_part_for_tricking_scatter, B_cols, n2 * n3 / dims[0], MPI_DOUBLE, 0, row_comm);
        //режем матрицу В, используя созданных "обманны" тип

        //рассылаем матрицы А и В по строкам и колонкам соответственно
        MPI_Type_free(&b_part_vec);
        MPI_Type_free(&b_part_for_tricking_scatter);
    }
    MPI_Bcast(A_rows, n1 * n2 / dims[1], MPI_DOUBLE, 0, row_comm);
    MPI_Bcast(B_cols, n2 * n3 / dims[0], MPI_DOUBLE, 0, col_comm);
    //-----------------

    //умножаем пришедшие на процессор части матриц
    int A_part_rows = n1 / dims[1];
    int B_part_cols = n3 / dims[0];
    double C_part[B_part_cols * A_part_rows];

    for(int row_A = 0; row_A < A_part_rows; row_A++) {
        for(int col_B = 0; col_B < B_part_cols; col_B++) {
            C_part[row_A * n2 + col_B] = 0;
            for(int i = 0; i < n2; i++) {
                C_part[row_A * n2 + col_B] += A_rows[row_A * n2 + i] * B_cols[col_B * n2 + i];
            }
        }
    }
    //--------------------------------------------

    MPI_Datatype recieve_block;
    MPI_Datatype recieve_block_resized;
    MPI_Type_vector( A_part_rows, B_part_cols, n3 , MPI_DOUBLE , &recieve_block);// тип для минора матрицы С
    MPI_Type_commit(&recieve_block);
    MPI_Type_create_resized(recieve_block, 0, sizeof(double) * B_part_cols, &recieve_block_resized);
    MPI_Type_commit(&recieve_block_resized);

    int displs[world_size];
    int recvcounts[world_size];
    if(world_rank == 0) {
        int displs_coords[2];
        for(int i = 0; i < world_size; i++) {
            MPI_Cart_coords(table_comm, i, 2, displs_coords);
            printf("My cart coords are (%d , %d)", displs_coords[0], displs_coords[1]);
            displs[i] = n1 / dims[1] * dims[0] * displs_coords[1] + displs_coords[0];

            recvcounts[i] = 1;
        }
    }
    MPI_Datatype send_vec;
    MPI_Type_contiguous(n1 * n3 / (dims[0] * dims[1]), MPI_DOUBLE, &send_vec);
    MPI_Type_commit(&send_vec);
    MPI_Gatherv( C_part , 1 , send_vec , AB , recvcounts , displs , recieve_block_resized , 0, table_comm);
    end = MPI_Wtime();
    if(world_rank == 0) {
        cout << "Time taken: " << end - start << endl;
    }
    MPI_Finalize();

}
