#include "v6.cu"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void check(CUresult status) {
  if (status != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(status, &message);
    throw std::runtime_error(message ? message : "CUDA driver error");
  }
}

void check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS error " + std::to_string(int(status)));
}

struct Options {
  int rows = 4096;
  int columns = 4096;
  int reduction = 4096;
  int warmup = 100;
  int iterations = 100;
  int rounds = 7;
  int graph_warmup = 50;
  bool correctness_only = false;
};

Options parse_options(int argc, char** argv) {
  Options options;
  std::vector<int> dimensions;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--correctness-only") {
      options.correctness_only = true;
    } else {
      size_t consumed = 0;
      const int value = std::stoi(argument, &consumed);
      if (consumed != argument.size() || value <= 0)
        throw std::runtime_error("Dimensions must be positive integers");
      dimensions.push_back(value);
    }
  }
  if (!dimensions.empty()) {
    if (dimensions.size() != 3) throw std::runtime_error("Usage: benchmark [--correctness-only] [M N K]");
    options.rows = dimensions[0];
    options.columns = dimensions[1];
    options.reduction = dimensions[2];
  }
  if (options.rows % mojo_v6::MMA_M || options.columns % mojo_v6::MMA_N ||
      options.reduction % mojo_v6::BLOCK_K)
    throw std::runtime_error("Required: M % 256 == 0, N % " + std::to_string(mojo_v6::MMA_N) + " == 0, K % 64 == 0");
  return options;
}

struct DeviceBuffer {
  __nv_bfloat16* data = nullptr;
  explicit DeviceBuffer(size_t elements) { check(cudaMalloc(&data, elements * sizeof(*data))); }
  ~DeviceBuffer() { cudaFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

template <typename Launch>
cudaGraphExec_t capture(Launch launch, cudaStream_t stream, int iterations) {
  cudaGraph_t graph;
  cudaGraphExec_t executable;
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  for (int iteration = 0; iteration < iterations; ++iteration) launch();
  check(cudaStreamEndCapture(stream, &graph));
  check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  check(cudaGraphDestroy(graph));
  return executable;
}

double time_graph(cudaGraphExec_t executable, cudaStream_t stream,
                  cudaEvent_t start, cudaEvent_t stop, int iterations) {
  check(cudaEventRecord(start, stream));
  check(cudaGraphLaunch(executable, stream));
  check(cudaEventRecord(stop, stream));
  check(cudaEventSynchronize(stop));
  float milliseconds = 0;
  check(cudaEventElapsedTime(&milliseconds, start, stop));
  return milliseconds / iterations;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void print_samples(const char* name, const std::vector<double>& samples) {
  std::printf("\"%s\":[", name);
  for (size_t index = 0; index < samples.size(); ++index)
    std::printf("%s%.9f", index ? "," : "", samples[index]);
  std::printf("]");
}

int run(const Options& options) {
  cudaDeviceProp properties;
  check(cudaGetDeviceProperties(&properties, 0));
  if (properties.major != 10 || properties.minor != 0)
    throw std::runtime_error("This kernel requires an sm_100a Blackwell GPU");
  int runtime_version = 0;
  int driver_version = 0;
  check(cudaRuntimeGetVersion(&runtime_version));
  check(cudaDriverGetVersion(&driver_version));
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cublasHandle_t handle;
  check(cublasCreate(&handle));
  check(cublasSetStream(handle, stream));
  check(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
  int cublas_version = 0;
  check(cublasGetVersion(handle, &cublas_version));
  std::printf("GPU=%s SMs=%d CUDA_runtime=%d driver=%d cuBLAS=%d\n",
              properties.name, properties.multiProcessorCount, runtime_version, driver_version, cublas_version);
  std::printf("M=%d N=%d K=%d; BF16 A/B/C, FP32 accumulation; tile=128x%dx64 per CTA; MMA=256x%dx16; cluster=2x1x1; 128B swizzle; warp-specialized pipeline; stmatrix + TMA output; TMEM=512\n",
              options.rows, options.columns, options.reduction, mojo_v6::BLOCK_N, mojo_v6::MMA_N);
  std::printf("AB stages=%d shared_bytes=%d\n", mojo_v6::AB_STAGES, mojo_v6::SHARED_BYTES);

  const size_t a_count = size_t(options.rows) * options.reduction;
  const size_t b_count = size_t(options.columns) * options.reduction;
  const size_t c_count = size_t(options.rows) * options.columns;
  std::vector<__nv_bfloat16> host_a(a_count), host_b(b_count), host_c(c_count), host_reference(c_count);
  DeviceBuffer device_a(a_count), device_b(b_count), device_c(c_count), reference(c_count);
  alignas(64) CUtensorMap a_map, b_map, c_map;
  check(mojo_v6::encode_tensor_map(&a_map, device_a.data, options.rows, options.reduction, mojo_v6::BLOCK_M));
  check(mojo_v6::encode_tensor_map(&b_map, device_b.data, options.columns, options.reduction, mojo_v6::BLOCK_N));
  check(mojo_v6::encode_output(&c_map, device_c.data, options.rows, options.columns));
  check(cudaFuncSetAttribute(mojo_v6::gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, mojo_v6::SHARED_BYTES));
  const float alpha = 1;
  const float beta = 0;
  auto launch_custom = [&]() {
    mojo_v6::launch(a_map, b_map, c_map, options.rows, options.columns, options.reduction, stream);
  };
  auto launch_cublas = [&]() {
    check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                       options.columns, options.rows, options.reduction,
                       &alpha, device_b.data, CUDA_R_16BF, options.reduction,
                       device_a.data, CUDA_R_16BF, options.reduction,
                       &beta, reference.data, CUDA_R_16BF, options.columns,
                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  };
  auto upload = [&]() {
    check(cudaMemcpyAsync(device_a.data, host_a.data(), a_count * 2, cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(device_b.data, host_b.data(), b_count * 2, cudaMemcpyHostToDevice, stream));
    check(cudaMemsetAsync(device_c.data, 0xff, c_count * 2, stream));
    check(cudaMemsetAsync(reference.data, 0xff, c_count * 2, stream));
  };
  auto download = [&]() {
    check(cudaMemcpyAsync(host_c.data(), device_c.data, c_count * 2, cudaMemcpyDeviceToHost, stream));
    check(cudaMemcpyAsync(host_reference.data(), reference.data, c_count * 2, cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
  };

  for (size_t index = 0; index < a_count; ++index)
    host_a[index] = __float2bfloat16_rn(float(index % options.reduction));
  for (int row = 0; row < options.columns; ++row)
    for (int column = 0; column < options.reduction; ++column)
      host_b[size_t(row) * options.reduction + column] = __float2bfloat16_rn(row == column ? 1.f : 0.f);
  upload();
  launch_custom();
  check(cudaGetLastError());
  launch_cublas();
  download();
  for (size_t index = 0; index < c_count; ++index) {
    const int column = index % options.columns;
    const float expected = column < options.reduction ? __bfloat162float(host_a[column]) : 0;
    if (__bfloat162float(host_c[index]) != expected || __bfloat162float(host_reference[index]) != expected)
      throw std::runtime_error("Identity correctness failed at " + std::to_string(index) +
                               ": actual=" + std::to_string(__bfloat162float(host_c[index])) +
                               " expected=" + std::to_string(expected));
  }
  std::puts("PASS: original Mojo identity input, exact comparison (CUDA and cuBLAS)");

  std::mt19937 generator(20260918);
  std::uniform_real_distribution<float> distribution(-1, 1);
  for (auto& value : host_a) value = __float2bfloat16_rn(distribution(generator));
  for (auto& value : host_b) value = __float2bfloat16_rn(distribution(generator));
  upload();
  launch_custom();
  check(cudaGetLastError());
  launch_cublas();
  download();
  double max_error = 0;
  double relative_l2 = 0;
  auto validate_random = [&]() {
    double error_squared = 0;
    double reference_squared = 0;
    max_error = 0;
    for (size_t index = 0; index < c_count; ++index) {
      const double actual = __bfloat162float(host_c[index]);
      const double expected = __bfloat162float(host_reference[index]);
      const double error = std::abs(actual - expected);
      if (!std::isfinite(actual) || !std::isfinite(expected) || error > 0.03125 + 0.01 * std::abs(expected))
        throw std::runtime_error("Random correctness failed at " + std::to_string(index) +
                                 ": actual=" + std::to_string(actual) + " reference=" + std::to_string(expected));
      max_error = std::max(max_error, error);
      error_squared += error * error;
      reference_squared += expected * expected;
    }
    relative_l2 = std::sqrt(error_squared / std::max(reference_squared, 1e-30));
    if (relative_l2 > 0.005) throw std::runtime_error("Relative L2 error too large");
  };
  validate_random();
  for (int sample = 0; sample < 32; ++sample) {
    const int row = generator() % options.rows;
    const int column = generator() % options.columns;
    double expected = 0;
    for (int reduction = 0; reduction < options.reduction; ++reduction)
      expected += double(__bfloat162float(host_a[size_t(row) * options.reduction + reduction])) *
                  double(__bfloat162float(host_b[size_t(column) * options.reduction + reduction]));
    expected = __bfloat162float(__float2bfloat16_rn(float(expected)));
    const double actual = __bfloat162float(host_c[size_t(row) * options.columns + column]);
    if (std::abs(actual - expected) > 0.03125 + 0.01 * std::abs(expected))
      throw std::runtime_error("CPU FP64 sampled reference failed");
  }
  std::printf("PASS: dense random input vs cuBLAS, max_abs_error=%.9g relative_l2=%.9g; CPU FP64 samples=32\n",
              max_error, relative_l2);

  if (!options.correctness_only) {
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
      launch_custom();
      launch_cublas();
    }
    check(cudaStreamSynchronize(stream));
    auto custom_graph = capture(launch_custom, stream, options.iterations);
    auto cublas_graph = capture(launch_cublas, stream, options.iterations);
    for (int replay = 0; replay < options.graph_warmup; ++replay) {
      check(cudaGraphLaunch(custom_graph, stream));
      check(cudaGraphLaunch(cublas_graph, stream));
    }
    check(cudaStreamSynchronize(stream));
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    std::vector<double> custom_times, cublas_times;
    for (int round = 0; round < options.rounds; ++round) {
      if (round % 2 == 0) {
        custom_times.push_back(time_graph(custom_graph, stream, start, stop, options.iterations));
        cublas_times.push_back(time_graph(cublas_graph, stream, start, stop, options.iterations));
      } else {
        cublas_times.push_back(time_graph(cublas_graph, stream, start, stop, options.iterations));
        custom_times.push_back(time_graph(custom_graph, stream, start, stop, options.iterations));
      }
    }
    download();
    validate_random();
    const double custom_ms = median(custom_times);
    const double cublas_ms = median(cublas_times);
    const double operations = 2.0 * options.rows * options.columns * options.reduction;
    std::printf("CUDA v6: %.6f ms, %.2f TFLOPS\n", custom_ms, operations / custom_ms / 1e9);
    std::printf("cuBLAS:  %.6f ms, %.2f TFLOPS\n", cublas_ms, operations / cublas_ms / 1e9);
    std::printf("CUDA/cuBLAS throughput: %.2f%%; CUDA latency / cuBLAS: %.3fx\n", 100 * cublas_ms / custom_ms, custom_ms / cublas_ms);
    std::printf("RESULT_JSON={\"gpu\":\"%s\",\"sms\":%d,\"cuda_runtime\":%d,\"driver\":%d,\"cublas_version\":%d,",
                properties.name, properties.multiProcessorCount, runtime_version, driver_version, cublas_version);
    std::printf("\"m\":%d,\"n\":%d,\"k\":%d,\"warmup\":%d,\"iterations\":%d,\"rounds\":%d,",
                options.rows, options.columns, options.reduction, options.warmup, options.iterations, options.rounds);
    std::printf("\"mma_n\":%d,\"ab_stages\":%d,\"shared_bytes\":%d,", mojo_v6::MMA_N, mojo_v6::AB_STAGES, mojo_v6::SHARED_BYTES);
    std::printf("\"graph_warmup_replays\":%d,", options.graph_warmup);
    std::printf("\"custom_ms\":%.9f,\"cublas_ms\":%.9f,\"custom_tflops\":%.6f,\"cublas_tflops\":%.6f,\"throughput_percent\":%.6f,",
                custom_ms, cublas_ms, operations / custom_ms / 1e9, operations / cublas_ms / 1e9, 100 * cublas_ms / custom_ms);
    std::printf("\"max_abs_error\":%.9g,\"relative_l2\":%.9g,", max_error, relative_l2);
    print_samples("custom_ms_samples", custom_times);
    std::printf(",");
    print_samples("cublas_ms_samples", cublas_times);
    std::printf("}\n");
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaGraphExecDestroy(custom_graph));
    check(cudaGraphExecDestroy(cublas_graph));
  }
  check(cublasDestroy(handle));
  check(cudaStreamDestroy(stream));
  return 0;
}

int main(int argc, char** argv) {
  try {
    return run(parse_options(argc, argv));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
  }
}
