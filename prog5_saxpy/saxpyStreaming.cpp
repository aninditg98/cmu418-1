#include <smmintrin.h> // For _mm_stream_load_si128
#include <emmintrin.h> // For _mm_mul_ps
#include <assert.h>
#include <stdint.h>

extern void saxpySerial(int N,
			float scale,
			float X[],
			float Y[],
			float result[]);


void saxpyStreaming(int N,
                    float scale,
                    float X[],
                    float Y[],
                    float result[])
{
    __m128 scale_vec = _mm_set1_ps(scale);

    for (int i = 0; i < N; i += 1) {
        __m128i result_int_x = _mm_stream_load_si128((__m128i*)X + i);
        __m128 result_float_x = _mm_castsi128_ps(result_int_x);

        __m128i result_int_y = _mm_stream_load_si128((__m128i*)Y + i);
        __m128 result_float_y = _mm_castsi128_ps(result_int_y);

        __m128 result_float = _mm_add_ps(_mm_mul_ps(scale_vec, result_float_x), result_float_y);
        _mm_stream_ps(result + i, result_float);

    }
}

