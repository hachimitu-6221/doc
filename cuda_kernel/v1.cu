#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace mojo_v1 {

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_K = 64;
constexpr int THREADS = 128;
constexpr int TMEM_COLUMNS = 512;
constexpr int A_BYTES = BLOCK_M * BLOCK_K * 2;
constexpr int B_BYTES = BLOCK_N * BLOCK_K * 2;
constexpr int SHARED_BYTES = A_BYTES + B_BYTES + 24;

__device__ __forceinline__ void wait_barrier(uint32_t address, uint32_t phase) {
  asm volatile(
      "{ .reg .pred ready; wait_loop: "
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%0], %1; "
      "@!ready bra wait_loop; }"
      :: "r"(address), "r"(phase) : "memory");
}

__device__ __forceinline__ void load_tile(
    uint32_t destination, const CUtensorMap* tensor_map,
    int row, int reduction_tile, uint32_t barrier) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {0, %2, %3}], [%4];"
      :: "r"(destination), "l"(tensor_map), "r"(row),
         "r"(reduction_tile), "r"(barrier) : "memory");
}

__device__ __forceinline__ uint64_t descriptor(uint32_t address, int rows) {
  return uint64_t(address >> 4) |
         (uint64_t(rows) << 16) |
         (uint64_t(8) << 32) |
         (uint64_t(1) << 46);
}

__global__ __launch_bounds__(THREADS)
void gemm(const __grid_constant__ CUtensorMap a_map,
          const __grid_constant__ CUtensorMap b_map,
          __nv_bfloat16* output, int columns, int reduction_iterations) {
  extern __shared__ __align__(128) unsigned char shared[];
  const uint32_t shared_base = uint32_t(__cvta_generic_to_shared(shared));
  const uint32_t tma_barrier = shared_base + A_BYTES + B_BYTES;
  const uint32_t mma_barrier = tma_barrier + 8;
  const uint32_t allocation_address = mma_barrier + 8;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;

  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(tma_barrier) : "memory");
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mma_barrier) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 0) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(allocation_address), "r"(TMEM_COLUMNS) : "memory");
  }
  __syncthreads();

  const uint32_t tmem = *reinterpret_cast<uint32_t*>(shared + A_BYTES + B_BYTES + 16);
  constexpr uint32_t instruction_descriptor =
      (1u << 4) | (1u << 7) | (1u << 10) |
      ((BLOCK_N / 8) << 17) | ((BLOCK_M / 16) << 24);
  uint32_t phase = 0;

  for (int iteration = 0; iteration < reduction_iterations; ++iteration) {
    if (threadIdx.x == 0) {
      asm volatile(
          "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
          :: "r"(tma_barrier), "r"(A_BYTES + B_BYTES) : "memory");
      load_tile(shared_base, &a_map, blockIdx.y * BLOCK_M, iteration * 8, tma_barrier);
      load_tile(shared_base + A_BYTES, &b_map, blockIdx.x * BLOCK_N,
                iteration * 8, tma_barrier);
    }
    wait_barrier(tma_barrier, phase);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    if (threadIdx.x == 0) {
#pragma unroll
      for (int step = 0; step < BLOCK_K / 16; ++step) {
        const uint64_t a_descriptor = descriptor(shared_base + step * BLOCK_M * 32, BLOCK_M);
        const uint64_t b_descriptor = descriptor(shared_base + A_BYTES + step * BLOCK_N * 32, BLOCK_N);
        const uint32_t accumulate = iteration != 0 || step != 0;
        asm volatile(
            "{ .reg .pred accumulate; setp.ne.u32 accumulate, %4, 0; "
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, accumulate; }"
            :: "r"(tmem), "l"(a_descriptor), "l"(b_descriptor),
               "r"(instruction_descriptor), "r"(accumulate) : "memory");
      }
      asm volatile(
          "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
          :: "r"(mma_barrier) : "memory");
    }
    wait_barrier(mma_barrier, phase);
    phase ^= 1;
  }

  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  float fragment[128];
  const uint32_t read_address = tmem + (uint32_t(warp * 32) << 16);
  asm volatile(
      "tcgen05.ld.sync.aligned.16x256b.x32.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
      "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
      "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, "
      "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
      : "=f"(fragment[0]), "=f"(fragment[1]), "=f"(fragment[2]), "=f"(fragment[3]),
        "=f"(fragment[4]), "=f"(fragment[5]), "=f"(fragment[6]), "=f"(fragment[7]),
        "=f"(fragment[8]), "=f"(fragment[9]), "=f"(fragment[10]), "=f"(fragment[11]),
        "=f"(fragment[12]), "=f"(fragment[13]), "=f"(fragment[14]), "=f"(fragment[15]),
        "=f"(fragment[16]), "=f"(fragment[17]), "=f"(fragment[18]), "=f"(fragment[19]),
        "=f"(fragment[20]), "=f"(fragment[21]), "=f"(fragment[22]), "=f"(fragment[23]),
        "=f"(fragment[24]), "=f"(fragment[25]), "=f"(fragment[26]), "=f"(fragment[27]),
        "=f"(fragment[28]), "=f"(fragment[29]), "=f"(fragment[30]), "=f"(fragment[31]),
        "=f"(fragment[32]), "=f"(fragment[33]), "=f"(fragment[34]), "=f"(fragment[35]),
        "=f"(fragment[36]), "=f"(fragment[37]), "=f"(fragment[38]), "=f"(fragment[39]),
        "=f"(fragment[40]), "=f"(fragment[41]), "=f"(fragment[42]), "=f"(fragment[43]),
        "=f"(fragment[44]), "=f"(fragment[45]), "=f"(fragment[46]), "=f"(fragment[47]),
        "=f"(fragment[48]), "=f"(fragment[49]), "=f"(fragment[50]), "=f"(fragment[51]),
        "=f"(fragment[52]), "=f"(fragment[53]), "=f"(fragment[54]), "=f"(fragment[55]),
        "=f"(fragment[56]), "=f"(fragment[57]), "=f"(fragment[58]), "=f"(fragment[59]),
        "=f"(fragment[60]), "=f"(fragment[61]), "=f"(fragment[62]), "=f"(fragment[63]),
        "=f"(fragment[64]), "=f"(fragment[65]), "=f"(fragment[66]), "=f"(fragment[67]),
        "=f"(fragment[68]), "=f"(fragment[69]), "=f"(fragment[70]), "=f"(fragment[71]),
        "=f"(fragment[72]), "=f"(fragment[73]), "=f"(fragment[74]), "=f"(fragment[75]),
        "=f"(fragment[76]), "=f"(fragment[77]), "=f"(fragment[78]), "=f"(fragment[79]),
        "=f"(fragment[80]), "=f"(fragment[81]), "=f"(fragment[82]), "=f"(fragment[83]),
        "=f"(fragment[84]), "=f"(fragment[85]), "=f"(fragment[86]), "=f"(fragment[87]),
        "=f"(fragment[88]), "=f"(fragment[89]), "=f"(fragment[90]), "=f"(fragment[91]),
        "=f"(fragment[92]), "=f"(fragment[93]), "=f"(fragment[94]), "=f"(fragment[95]),
        "=f"(fragment[96]), "=f"(fragment[97]), "=f"(fragment[98]), "=f"(fragment[99]),
        "=f"(fragment[100]), "=f"(fragment[101]), "=f"(fragment[102]), "=f"(fragment[103]),
        "=f"(fragment[104]), "=f"(fragment[105]), "=f"(fragment[106]), "=f"(fragment[107]),
        "=f"(fragment[108]), "=f"(fragment[109]), "=f"(fragment[110]), "=f"(fragment[111]),
        "=f"(fragment[112]), "=f"(fragment[113]), "=f"(fragment[114]), "=f"(fragment[115]),
        "=f"(fragment[116]), "=f"(fragment[117]), "=f"(fragment[118]), "=f"(fragment[119]),
        "=f"(fragment[120]), "=f"(fragment[121]), "=f"(fragment[122]), "=f"(fragment[123]),
        "=f"(fragment[124]), "=f"(fragment[125]), "=f"(fragment[126]), "=f"(fragment[127])
      : "r"(read_address));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  __syncthreads();

  if (warp == 0) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem), "r"(TMEM_COLUMNS) : "memory");
  }

#pragma unroll
  for (int column_tile = 0; column_tile < BLOCK_N / 8; ++column_tile) {
#pragma unroll
    for (int row_tile = 0; row_tile < 2; ++row_tile) {
      const int row = blockIdx.y * BLOCK_M + warp * 16 + lane / 4 + row_tile * 8;
      const int column = blockIdx.x * BLOCK_N + column_tile * 8 + (lane % 4) * 2;
      const int fragment_index = column_tile * 4 + row_tile * 2;
      *reinterpret_cast<__nv_bfloat162*>(output + int64_t(row) * columns + column) =
          __float22bfloat162_rn(make_float2(fragment[fragment_index], fragment[fragment_index + 1]));
    }
  }
}

CUresult encode_tensor_map(CUtensorMap* tensor_map, __nv_bfloat16* pointer,
                           int rows, int reduction, int tile_rows) {
  const uint64_t dimensions[] = {8, uint64_t(rows), uint64_t(reduction / 8)};
  const uint64_t strides[] = {uint64_t(reduction) * 2, 16};
  const uint32_t box[] = {8, uint32_t(tile_rows), 8};
  const uint32_t element_strides[] = {1, 1, 1};
  return cuTensorMapEncodeTiled(
      tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, pointer,
      dimensions, strides, box, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void launch(const CUtensorMap& a_map, const CUtensorMap& b_map,
            __nv_bfloat16* output, int rows, int columns, int reduction,
            cudaStream_t stream) {
  gemm<<<dim3(columns / BLOCK_N, rows / BLOCK_M), THREADS, SHARED_BYTES, stream>>>(
      a_map, b_map, output, columns, reduction / BLOCK_K);
}

}
