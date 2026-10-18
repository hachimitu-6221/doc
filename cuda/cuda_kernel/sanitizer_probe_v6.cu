#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

cudaError_t audit_custom_completion() {
  const cudaError_t launch_status = cudaGetLastError();
  std::puts("AUDIT: custom launched; waiting for completion");
  if (launch_status != cudaSuccess) return launch_status;
  const cudaError_t status = cudaDeviceSynchronize();
  std::printf("AUDIT: custom completed: %s\n", cudaGetErrorString(status));
  return status;
}

template <typename... Arguments>
cublasStatus_t audit_cublas(Arguments... arguments) {
  std::puts("AUDIT: cuBLAS launch");
  const cublasStatus_t status = cublasGemmEx(arguments...);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  const cudaError_t completion = cudaDeviceSynchronize();
  std::printf("AUDIT: cuBLAS completed: %s\n", cudaGetErrorString(completion));
  return completion == cudaSuccess ? status : CUBLAS_STATUS_EXECUTION_FAILED;
}

#define cudaGetLastError audit_custom_completion
#define cublasGemmEx audit_cublas
#define main benchmark_main
#include "benchmark_v6.cu"
#undef main
#undef cublasGemmEx
#undef cudaGetLastError

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  return benchmark_main(argc, argv);
}
