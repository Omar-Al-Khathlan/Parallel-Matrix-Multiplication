# include <stdio.h>
# include <stdlib.h>
# include <omp.h>
# define PAD 8


void InitilizeMatrix(int *matrix, int nbRow, int nbColumn)
{
    int i, j;
    for (i = 0; i < nbRow; i++)
    {
        for (j = 0; j < nbColumn; j++)
        {
            matrix[i*nbColumn+j] = 0;
        }
    }
}

void UnitMatrix(int *matrix, int nbRow, int nbColumn)
{
    int i, j;
    for (i = 0; i < nbRow; i++)
    {
        for (j = 0; j < nbColumn; j++)
        {
            if( i == j )
                matrix[i*nbColumn+j] = 1;
            else
                matrix[i*nbColumn+j] = 0;
        }
    }
}

void PrintMatrix(int* matrix, int nbRow, int nbColumn)
{
    int i, j;
    printf("[ ");
    for (i = 0; i < nbRow; i++)
    {
        for (j = 0; j < nbColumn; j++)
        {
            printf("%d, ", matrix[i*nbColumn+j]);
        }
        printf("\n  ");
    }
    printf("]\n\n");
}

void MultiplyMatrix(int* A, int* B, int nbRow, int nbColumn, int* C, int threadCount)
{
    
    omp_set_num_threads(threadCount);
    #pragma omp parallel
    {
        int num_thread = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int i, j, k;
        for (i = thread_id; i < nbRow; i += num_thread)
        {
            for (j = thread_id; j < nbColumn; j += num_thread)
            {
                C[i*nbColumn+j] = 0;
                for (k = thread_id; k < nbColumn; k += num_thread)
                {
                    C[i*nbColumn+j] += A[i*nbColumn+k] * B[k*nbColumn+j];
                }
            }
        }
    }
    // This is the parallel implementation.
    // int i, j, k;
    // for (i = 0; i < nbRow; i++)
    // {
    //     for (j = 0; j < nbColumn; j++)
    //     {
    //         C[i*nbColumn+j] = 0;
    //         for (k = 0; k < nbColumn; k++)
    //         {
    //             C[i*nbColumn+j] += A[i*nbColumn+k] * B[k*nbColumn+j];
    //         }
    //     }
    // }
}

int main()
{
    double time_spent = 0.0;
    int degree = 3;
    int threadCount = 1;
    printf("enter the array size: ");
    scanf("%d",&degree);
    int *A = malloc(sizeof(int) * degree * degree * PAD);
    UnitMatrix(A, degree, degree);
    int *B = malloc(sizeof(int) * degree * degree * PAD);
    UnitMatrix(B, degree, degree);

    printf("enter the thread count to be used: ");
    scanf("%d",&threadCount);

    printf("\nInput: \n");
    PrintMatrix(A, degree, degree);
    PrintMatrix(B, degree, degree);

    int *C = malloc(sizeof(int) * degree * degree * PAD);
    InitilizeMatrix(C, degree, degree);

    double begin = omp_get_wtime();

    MultiplyMatrix(A, B, degree, degree, C, threadCount);

    double end = omp_get_wtime();

    printf("\nOutput: \n");
    PrintMatrix(C, degree, degree);

    time_spent = end - begin;
    printf("Time elpased is %f seconds\n\n", time_spent);
    return 0;
}
