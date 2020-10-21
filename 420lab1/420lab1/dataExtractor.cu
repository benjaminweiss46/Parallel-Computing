
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dataExtractor.h"


void showHolders(struct container * arr, int size) {
    int A;
    int B;
    int op;
    int  result;
    for (int i = 0; i < size; i++) {
        A = arr[i].A;
        B = arr[i].B;
        op = arr[i].op;
        result = arr[i].result;
        printf("Holder %d,\nA: %d\nB: %d\nOperator: %d\nResult: %d\n\n", i, A, B, op, result);
    }
}
struct container * getCSVData(char* file, int size)
{
    FILE* my_file = fopen(file, "r");
    if (my_file == NULL) {
        return 0;
    }
    char line[10];
    int count = 0; 
    struct container * holders = (struct container *)malloc(sizeof(struct container)*size); 
    char* tok;
    int A;
    int B;
    int op;
    while (fgets(line, 10, my_file))
    {
        char* tmp = strdup(line);
        //holders[count] = (struct container*)malloc(sizeof(struct container));
        tok = strtok(tmp, ",");
        holders[count].A = atoi(tok);
        tok = strtok(NULL, ",");
        holders[count].B = atoi(tok);
        tok = strtok(NULL, ",");
        holders[count].op = atoi(tok);
        holders[count].result = -1; 
        count++;
        if (count > size) {
            break;
        }
    }
    //showHolders(holders, size);
    return holders;
}
