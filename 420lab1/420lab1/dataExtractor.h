#pragma once
#ifndef DATAEXTRACTOR_H_ //guard
#define DATAEXTRACTOR_H_
struct container * getCSVData(char* file, int size);
void showHolders(struct container* arr, int size);
struct container {
	int A;
	int B;
	int op;
	int result;
};
#endif 