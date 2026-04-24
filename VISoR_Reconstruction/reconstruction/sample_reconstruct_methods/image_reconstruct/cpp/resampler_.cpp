#include "torch/extension.h"
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include "resampler.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor resample_affine(torch::Tensor src, torch::Tensor dst, torch::Tensor transform)
{
	CHECK_CUDA(src)
    auto t1 = std::chrono::high_resolution_clock::now();

	auto src_p = src.data();
	unsigned int src_shape[3] = { src.shape()[2],  src.shape()[1] , src.shape()[0] };

	auto dst_p = dst.mutable_data();
	unsigned int dst_shape[3] = { dst.shape()[2],  dst.shape()[1] , dst.shape()[0] };

	cudaError_t cudaStatus = _resample_affine(dst_p, dst_shape, src_p, src_shape, transform);
	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << "f() took "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
		<< " milliseconds\n";

	cudaStatus = cudaDeviceReset();
	
	return dst;
}

PYBIND11_MODULE(resampler_, m) {
	m.doc() = "GPU Resampler";
	m.def("resample_affine", &resample_affine, "GPU affine transform");
}

