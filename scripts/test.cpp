#include <omp.h>
#include <stdio.h>

int main() {
#pragma omp parallel
    {
        printf("tid=%d\n", omp_get_thread_num());
    }
}
