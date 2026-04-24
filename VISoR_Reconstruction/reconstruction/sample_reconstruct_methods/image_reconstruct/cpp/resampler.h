#pragma once
#include "cuda_runtime.h"

cudaError_t _resample_affine(uint16_t *dst, unsigned int* dst_size, const uint16_t *src, unsigned int* src_size, float* matrix);
