
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include "helper_cuda.h"
#include "resampler.h"

__device__ float linear_window(float x, float y, float z,
	float ww, float wh, float wd)
{
	float w = fmin(fmin(fmin(fmax(ww - abs(x), 0.f), fmax(wh - abs(y), 0.f)), fmax(wd - abs(z), 0.f)), 1.f);
	return w;
}


__global__ void resample_affine_kernel(cudaPitchedPtr dst, 
	unsigned int dst_w, unsigned int dst_h, unsigned int dst_d, 
	cudaPitchedPtr src, 
	unsigned int src_w, unsigned int src_h, unsigned int src_d,
	float* m, float window_shift_x, float window_shift_y, 
	float ww, float wh, float wd)
{
	uint16_t* p_dst = (uint16_t *)dst.ptr;
	uint16_t* p_src = (uint16_t *)src.ptr;
	int dst_pitch_y = dst.pitch / sizeof(uint16_t);
	int dst_pitch_z = dst_pitch_y * dst.ysize;
	int src_pitch_y = src.pitch / sizeof(uint16_t);
	int src_pitch_z = src_pitch_y * src.ysize;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	float x_src = x * m[0] + y * m[1] + z * m[2] + m[3];
	float y_src = x * m[4] + y * m[5] + z * m[6] + m[7];
	float z_src = x * m[8] + y * m[9] + z * m[10] + m[11];
	size_t n_dst = x + y * dst_pitch_y + z * dst_pitch_z;
	
	float value = 0;
	float weight = 0;
	for (int k = ceil(-wd + z_src); k <= floor(wd + z_src); ++k) {
		int y_start = ceil(-wh + y_src + window_shift_y * (float(k) - z_src));
		int y_end = floor(wh + y_src + window_shift_y * (float(k) - z_src));
		float Dz = (float)k - z_src;
		for (int j = y_start; j <= y_end; ++j) {
			int x_start = ceil(-ww + x_src + window_shift_x * (float(k) - z_src));
			int x_end = floor(ww + x_src + window_shift_x * (float(k) - z_src));
			float Dy = (float)j - y_src - window_shift_y * (float(k) - z_src);
			for (int i = x_start; i <= x_end; ++i) {
				size_t n_src = i + j * src_pitch_y + k * src_pitch_z;
				float Dx = (float)i - x_src - window_shift_x * (float(k) - z_src);
				float w = linear_window(Dx, Dy, Dz, ww, wh, wd);
				w = (0 <= i && i < src_w ? w : 0);
				w = (0 <= j && j < src_h ? w : 0);
				w = (0 <= k && k < src_d ? w : 0);
				n_src = (w == 0 ? 0 : n_src);
				weight += w;
				value += w * (float)p_src[n_src];
			}
		}
	}
	weight = (weight == 0 ? 1 : weight);

	p_dst[n_dst] = (uint16_t)(value / weight);
}

cudaError_t upload_3d_image(cudaPitchedPtr *dst, const uint16_t *src, unsigned int* size)
{
	cudaExtent ext = make_cudaExtent(size[0] * sizeof(uint16_t), size[1], size[2]);
	checkCudaErrors(cudaMalloc3D(dst, ext));
	cudaMemcpy3DParms p = { 0 };
	p.srcPtr.ptr = (void*)src;
	p.srcPtr.pitch = size[0] * sizeof(uint16_t);
	p.srcPtr.xsize = size[0];
	p.srcPtr.ysize = size[1];
	p.dstPtr.ptr = dst->ptr;
	p.dstPtr.pitch = dst->pitch;
	p.dstPtr.xsize = size[0];
	p.dstPtr.ysize = size[1];
	p.extent = ext;
	p.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&p));
	return cudaGetLastError();
}

cudaError_t download_3d_image(uint16_t *dst, cudaPitchedPtr *src, unsigned int* size)
{
	cudaExtent ext = make_cudaExtent(size[0] * sizeof(uint16_t), size[1], size[2]);
	cudaMemcpy3DParms p = { 0 };
	p.srcPtr.ptr = src->ptr;
	p.srcPtr.pitch = src->pitch;
	p.srcPtr.xsize = size[0];
	p.srcPtr.ysize = size[1];
	p.dstPtr.ptr = dst;
	p.dstPtr.pitch = size[0] * sizeof(uint16_t);
	p.dstPtr.xsize = size[0];
	p.dstPtr.ysize = size[1];
	p.extent = ext;
	p.kind = cudaMemcpyDeviceToHost;
	checkCudaErrors(cudaMemcpy3D(&p));
	return cudaGetLastError();
}
/*
float* invert_affine_core_mat(float* m) 
{
	float det = m[0] * (m[5] * m[10] - m[9] * m[6]) -
		m[1] * (m[4] * m[10] - m[6] * m[8]) +
		m[2] * (m[4] * m[9] - m[5] * m[8]);

	double invdet = 1 / det;

	float minv[9]; // inverse of matrix m
	minv[0] = (m[5] * m[10] - m[9] * m[6]) * invdet;
	minv[1] = (m[2] * m[9] - m[1] * m[10]) * invdet;
	minv[2] = (m[1] * m[6] - m[2] * m[5]) * invdet;
	minv[3] = (m[6] * m[8] - m[4] * m[10]) * invdet;
	minv[4] = (m[0] * m[10] - m[2] * m[8]) * invdet;
	minv[5] = (m[4] * m[2] - m[0] * m[6]) * invdet;
	minv[6] = (m[4] * m[9] - m[8] * m[5]) * invdet;
	minv[7] = (m[8] * m[1] - m[0] * m[9]) * invdet;
	minv[8] = (m[0] * m[5] - m[4] * m[1]) * invdet;
	return minv;
}*/

cudaError_t _resample_affine(uint16_t *dst, unsigned int* dst_size, const uint16_t *src, unsigned int* src_size, float* m)
{
	cudaPitchedPtr dev_dst;
	cudaPitchedPtr dev_src;
	
	float* dev_matrix;
	checkCudaErrors(cudaMalloc<float>(&dev_matrix, 12 * sizeof(float)));
	checkCudaErrors(cudaMemcpy(dev_matrix, m, 12 * sizeof(float), cudaMemcpyHostToDevice));
	float window_shift_x = (m[0] * m[8] + m[1] * m[9] + m[2] * m[10]) / (m[8] * m[8] + m[9] * m[9] + m[10] * m[10]);
	float window_shift_y = (m[4] * m[8] + m[5] * m[9] + m[6] * m[10]) / (m[8] * m[8] + m[9] * m[9] + m[10] * m[10]);
	float v_shifted_x[3] = {m[0] + m[8] * window_shift_x, m[1] + m[9] * window_shift_x, m[2] + m[10] * window_shift_x};
	float v_shifted_y[3] = {m[4] + m[8] * window_shift_y, m[5] + m[9] * window_shift_y, m[6] + m[10] * window_shift_y};
	float window_size_x = sqrt(pow(v_shifted_x[0], 2) + pow(v_shifted_x[1], 2) + pow(v_shifted_x[2], 2));
	float window_size_y = sqrt(pow(v_shifted_y[0], 2) + pow(v_shifted_y[1], 2) + pow(v_shifted_y[2], 2));
	float window_size_z = sqrt(m[8] * m[8] + m[9] * m[9] + m[10] * m[10]);
	window_size_x = fmax(window_size_x / 2, 1);
	window_size_y = fmax(window_size_y / 2, 1);
	window_size_z = fmax(window_size_z / 2, 1);
	//printf("%f %f %f %f %f", window_shift_x, window_shift_y, window_size_x, window_size_y, window_size_z);

	checkCudaErrors(cudaSetDevice(0));
	upload_3d_image(&dev_src, src, src_size);

	cudaExtent ext_dst = make_cudaExtent(dst_size[0] * sizeof(uint16_t), dst_size[1], dst_size[2]);
	checkCudaErrors(cudaMalloc3D(&dev_dst, ext_dst));

	dim3 num_threads(8, 8, 8);
	dim3 num_blocks(ceil(dst_size[0] / num_threads.x), ceil(dst_size[1] / num_threads.y), ceil(dst_size[2] / num_threads.z));
	resample_affine_kernel <<< num_blocks, num_threads>>> (dev_dst, 
		dst_size[0], dst_size[1], dst_size[2], 
		dev_src, 
		src_size[0], src_size[1], src_size[2],
		dev_matrix, window_shift_x, window_shift_y, 
		window_size_x, window_size_y, window_size_z);
	checkCudaErrors(cudaGetLastError());	
	checkCudaErrors(cudaDeviceSynchronize());

	download_3d_image(dst, &dev_dst, dst_size);
    return cudaGetLastError();
}
