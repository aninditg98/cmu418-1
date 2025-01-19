#include <stdio.h>
#include <algorithm>
#include <math.h>
#include "CMU418intrin.h"
#include "logger.h"
using namespace std;

void absSerial(float *values, float *output, int N)
{
	for (int i = 0; i < N; i++)
	{
		float x = values[i];
		if (x < 0)
		{
			output[i] = -x;
		}
		else
		{
			output[i] = x;
		}
	}
}

// implementation of absolute value using 15418 instrinsics
void absVector(float *values, float *output, int N)
{
	__cmu418_vec_float x;
	__cmu418_vec_float result;
	__cmu418_vec_float zero = _cmu418_vset_float(0.f);
	__cmu418_mask maskAll, maskIsNegative, maskIsNotNegative;

	//  Note: Take a careful look at this loop indexing.  This example
	//  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
	//  Why is that the case?
	for (int i = 0; i < N; i += VECTOR_WIDTH)
	{

		// All ones
		maskAll = _cmu418_init_ones();

		// All zeros
		maskIsNegative = _cmu418_init_ones(0);

		// Load vector of values from contiguous memory addresses
		_cmu418_vload_float(x, values + i, maskAll); // x = values[i];

		// Set mask according to predicate
		_cmu418_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

		// Execute instruction using mask ("if" clause)
		_cmu418_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

		// Inverse maskIsNegative to generate "else" mask
		maskIsNotNegative = _cmu418_mask_not(maskIsNegative); // } else {

		// Execute instruction ("else" clause)
		_cmu418_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

		// Write results back to memory
		_cmu418_vstore_float(output + i, result, maskAll);
	}
}

// Accepts an array of values and an array of exponents
// For each element, compute values[i]^exponents[i] and clamp value to
// 4.18.  Store result in outputs.
// Uses iterative squaring, so that total iterations is proportional
// to the log_2 of the exponent
void clampedExpSerial(float *values, int *exponents, float *output, int N)
{
	for (int i = 0; i < N; i++)
	{
		float x = values[i];
		float result = 1.f;
		int y = exponents[i];
		float xpower = x;
		while (y > 0)
		{
			if (y & 0x1)
			{
				result *= xpower;
			}
			xpower = xpower * xpower;
			y >>= 1;
		}
		if (result > 4.18f)
		{
			result = 4.18f;
		}
		output[i] = result;
	}
}

void printFloatVec(__cmu418_vec_float x)
{
	for (int j = 0; j < VECTOR_WIDTH; j++)
	{
		printf("%f ", x.value[j]);
	}
	printf("\n");
}

void printIntVec(__cmu418_vec_int x)
{
	for (int j = 0; j < VECTOR_WIDTH; j++)
	{
		printf("%d ", x.value[j]);
	}
	printf("\n");
}

void printMask(__cmu418_mask x)
{
	for (int j = 0; j < VECTOR_WIDTH; j++)
	{
		printf("%d ", x.value[j]);
	}
	printf("\n");
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
	printf("N is %d\n", N);

	int i = 0;
	__cmu418_vec_float x;
	__cmu418_vec_int y;
	__cmu418_vec_int zero = _cmu418_vset_int(0);
	__cmu418_mask maskAll, isZero, odds, evens, notZero, needsClamping;
	__cmu418_vec_int temp = _cmu418_vset_int(0);
	__cmu418_vec_int int_ones = _cmu418_vset_int(1);
	__cmu418_vec_float results, xpower;
	__cmu418_vec_float max_val = _cmu418_vset_float(4.18f);

	maskAll = _cmu418_init_ones();

	while (i < N)
	{
		_cmu418_vload_float(x, values + i, maskAll);	  // x = values[i];
		_cmu418_vload_float(xpower, values + i, maskAll); // x = values[i];

		_cmu418_vload_int(y, exponents + i, maskAll); // y = exponents[i];

		_cmu418_veq_int(isZero, y, zero, maskAll);
		results = _cmu418_vset_float(1.f);

		int rounds = 0;

		while (_cmu418_cntbits(isZero) < VECTOR_WIDTH)
		{
			_cmu418_vbitand_int(temp, int_ones, y, maskAll);
			_cmu418_veq_int(odds, temp, int_ones, maskAll);
			_cmu418_vmult_float(results, results, xpower, odds);
			notZero = _cmu418_mask_not(isZero);
			_cmu418_vmult_float(xpower, xpower, xpower, notZero);
			_cmu418_vshiftright_int(y, y, int_ones, notZero);
			rounds += 1;
			_cmu418_veq_int(isZero, y, zero, maskAll);
		}

		_cmu418_vgt_float(needsClamping, results, max_val, maskAll);
		_cmu418_vmove_float(results, max_val, needsClamping);

		
		__cmu418_mask toWrite = _cmu418_init_ones(min(N-i, VECTOR_WIDTH));
		_cmu418_vstore_float(output + i, results, toWrite);

		i += VECTOR_WIDTH;
	}
}

float arraySumSerial(float *values, int N)
{
	float sum = 0;
	for (int i = 0; i < N; i++)
	{
		sum += values[i];
	}

	return sum;
}

// Assume N % VECTOR_WIDTH == 0
// Assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

	float sum = 0;

	__cmu418_vec_float x;

	__cmu418_mask allOnes = _cmu418_init_ones(VECTOR_WIDTH);

	float iters = log(VECTOR_WIDTH) / log(2);


	for (int i = 0; i < N; i += VECTOR_WIDTH) {

		_cmu418_vload_float(x, values + i, allOnes);
		printFloatVec(x);
		for (float j = 0; j < iters; j++) {
			_cmu418_hadd_float(x, x);
			_cmu418_interleave_float(x, x);
			printFloatVec(x);
		}

		sum += x.value[0];

	}

	return sum;
}
