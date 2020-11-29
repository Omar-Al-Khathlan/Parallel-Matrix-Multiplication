# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>


int main ( int argc, char *argv[] );
double r8_mxm ( int l, int m, int n, int threadCount );
double r8_uniform_01 ( int *seed );


int main ( int argc, char *argv[] )
{
 
    printf ("Matrix multiplication tests.\n");
    
    int M[5] = {100, 200, 500, 1000, 2000};
    int P[4] = {1, 2, 3, 4};
    double time[5];

    int i, j, k, l;
    for (i = 0; i < 5; ++i)
    {
        printf("\nM = %d:\n", M[i]);
        for (j = 0; j < 4; ++j)
        {
            printf("P = %d, ", j+1);
            for (k = 0; k < 5; ++k)
            {
                time[k] = 0;
                time[k] = r8_mxm(M[i], M[i], M[i], P[j]);
                printf("Time%d = %lf, ", k+1, time[k]);
            }

            double sum = 0;
            for (l = 0; l < 5; ++l)
            {
                sum += time[l];
            }
            double avg = 0;
            avg = sum / 5;
            printf("Avg Time = %lf\n", avg);

        }

    }
    printf ("end of execution.\n");

    return 0;
}


double r8_mxm (int l, int m, int n, int threadCount)
{
    
    double *a;
    double *b;
    double *c;
    int i;
    int j;
    int k;
    int ops;
    double rate;
    int seed;
    double time_begin;
    double time_elapsed;
    double time_stop;
    
    a = ( double * ) malloc ( l * n * sizeof ( double ) );
    b = ( double * ) malloc ( l * m * sizeof ( double ) );
    c = ( double * ) malloc ( m * n * sizeof ( double ) );
    
    seed = 123456789;
    
    for ( k = 0; k < l * m; k++ )
    {
        b[k] = r8_uniform_01 ( &seed );
    }
    
    for ( k = 0; k < m * n; k++ )
    {
        c[k] = r8_uniform_01 ( &seed );
    }
    
    time_begin = omp_get_wtime();
    omp_set_num_threads(threadCount);
    # pragma omp parallel shared(a, b, c, l, m, n) private(i, j, k)
    # pragma omp for
    for ( j = 0; j < n; j++)
    {
        for ( i = 0; i < l; i++ )
        {
            a[i+j*l] = 0.0;
            for ( k = 0; k < m; k++ )
            {
            a[i+j*l] = a[i+j*l] + b[i+k*l] * c[k+j*m];
            
            }
        }
    }
    time_stop = omp_get_wtime();
    time_elapsed = time_stop - time_begin;
    free (a);
    free (b);
    free (c);

    return time_elapsed;
}


double r8_uniform_01 ( int *seed )
{
    int k;
 
    double r;
    
    k = *seed / 127773;
    
    *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;
    
    if ( *seed < 0 )
    {
        *seed = *seed + 2147483647;
    }
    
    r = ( double ) ( *seed ) * 4.656612875E-10;
    
    return r;
}