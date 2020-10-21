
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "dataExtractor.h"

/**
int operationCalc(int A, int B, int op) {
    if (op == 0) { // AND
        return A & B;
    }
    else if (op == 2) { // NAND
        return !(A * B);
    }
    else if (op == 1) { // OR
        return A | B;
    }
    else if (op == 3) { // NOR
        return !(A + B);
    }
    else if (op == 4) {  // XOR
        return A ^ B;
    }
    else { // XNOR
        return !(A ^ B);
    }
}

int main(int argc, char* argv[]) {
    // parse arguments
    if (argc == 4) {
        printf("Running sequential operators\n");
    }
    else if (argc > 4) {
        printf("Too many arguments given.\n");
        return -1;
    }
    else {
        printf("Missing arguments\n");
        return -1;
    }
    char* inputFile = argv[1];
    char* outputFile = argv[3];
    int size = atoi(argv[2]); 

    // get input data
    struct container * data = getCSVData(inputFile,size); //not defensive coding at all but no requirement specified
    
    // perform logic operations
    clock_t t1 = clock();
    for (int i = 0; i < size; i++) {
        data[i].result = operationCalc(data[i].A, data[i].B, data[i].op); //don't need result for sequential case
    }
    float t_op = (clock() - t1) * 1000.0 / CLOCKS_PER_SEC;

    // write results to output file
    char* newline = "\n";
    char* res;
    char buffer[10];
    FILE* solution = fopen(outputFile, "w");
    if (solution == NULL) {
        return -1;
    }
    for (int i = 0; i < size; i++) {
        res = itoa(data[i].result, buffer, 10);
        fprintf(solution, res);
        if (!(i == size - 1)) {
            fprintf(solution, newline);
        }
    }
    fclose(solution);

    printf("Logic operation execution time: %f ms\n", t_op);
    return 0;
}
**/