#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace mojo_v6 {

constexpr int BLOCK_M = 128;
constexpr int BLOCK_K = 64;
constexpr int MMA_M = 256;
constexpr int THREADS = 192;
constexpr int TMEM_COLUMNS = 512;
#ifndef MOJO_V6_MMA_N
#define MOJO_V6_MMA_N 256
#endif
constexpr int MMA_N = MOJO_V6_MMA_N;
static_assert(MMA_N == 128 || MMA_N == 256);
constexpr int BLOCK_N = MMA_N / 2;
constexpr int TMA_N = 32;
constexpr int C_STAGE_BYTES = BLOCK_M * TMA_N * 2;
constexpr int C_BYTES = 2 * C_STAGE_BYTES;

constexpr int A_BYTES = BLOCK_M * BLOCK_K * 2;
constexpr int B_BYTES = BLOCK_N * BLOCK_K * 2;
constexpr int AB_STAGES = (233472 - 1024 - C_BYTES - 12) / (A_BYTES + B_BYTES + 16);
constexpr int B_OFFSET = AB_STAGES * A_BYTES;
constexpr int C_OFFSET = B_OFFSET + AB_STAGES * B_BYTES;
constexpr int TMA_BARRIER = C_OFFSET + C_BYTES;
constexpr int MMA_BARRIER = TMA_BARRIER + AB_STAGES * 8;
constexpr int COMPUTE_BARRIER = MMA_BARRIER + AB_STAGES * 8;
constexpr int TMEM_ADDRESS = COMPUTE_BARRIER + 8;
constexpr int SHARED_BYTES = TMEM_ADDRESS + 4;

template <int Stages>
struct Pipeline {
  int index = 0;
  uint32_t phase = 0;
  __device__ void step() {
    if (++index == Stages) {
      index = 0;
      phase ^= 1;
    }
  }
};

__device__ __forceinline__ bool elect_one() {
  uint32_t result;
  asm volatile(
      "{ .reg .pred selected; elect.sync _|selected, 0xffffffff; "
      "selp.u32 %0, 1, 0, selected; }" : "=r"(result));
  return result != 0;
}

__device__ __forceinline__ uint32_t remote_address(uint32_t address, uint32_t rank) {
  uint32_t result;
  asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(result) : "r"(address), "r"(rank));
  return result;
}


__device__ __forceinline__ void wait_barrier(uint32_t address, uint32_t phase) {
  asm volatile(
      "{ .reg .pred ready; wait_loop: "
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%0], %1; "
      "@!ready bra wait_loop; }"
      :: "r"(address), "r"(phase) : "memory");
}

__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ void load_operand(uint32_t destination, const CUtensorMap* tensor_map,
                                            int row, int reduction_tile, uint32_t barrier, uint32_t rank) {
  const uint32_t leader_barrier = remote_address(barrier, 0);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2 "
      "[%0], [%1, {%3, %2}], [%4], %5;"
      :: "r"(destination), "l"(tensor_map), "r"(row), "r"(reduction_tile * BLOCK_K),
         "r"(leader_barrier), "h"(uint16_t(1u << rank)) : "memory");
}

__device__ __forceinline__ uint64_t descriptor(uint32_t address) {
  return uint64_t(address >> 4) | (uint64_t(64) << 32) |
         (uint64_t(1) << 46) | (uint64_t(2) << 61);
}

__device__ __forceinline__ void init_barrier(uint32_t address, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(address), "r"(count) : "memory");
}

__device__ __forceinline__ void commit_mma(uint32_t address) {
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
      :: "r"(address), "h"(uint16_t(3)) : "memory");
}

__device__ __forceinline__ void load_fragment(uint32_t address, float (&fragment)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.16x256b.x4.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
      : "=f"(fragment[0]), "=f"(fragment[1]), "=f"(fragment[2]), "=f"(fragment[3]),
        "=f"(fragment[4]), "=f"(fragment[5]), "=f"(fragment[6]), "=f"(fragment[7]),
        "=f"(fragment[8]), "=f"(fragment[9]), "=f"(fragment[10]), "=f"(fragment[11]),
        "=f"(fragment[12]), "=f"(fragment[13]), "=f"(fragment[14]), "=f"(fragment[15])
      : "r"(address));
}

__device__ __forceinline__ void store_fragment(uint32_t destination, const float (&fragment)[16], int lane) {
#pragma unroll
  for (int tile = 0; tile < 2; ++tile) {
    uint32_t packed[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const int index = tile * 8 + pair * 2;
      const __nv_bfloat162 value = __float22bfloat162_rn(make_float2(fragment[index], fragment[index + 1]));
      const __nv_bfloat162_raw raw = value;
      packed[pair] = uint32_t(raw.x) | (uint32_t(raw.y) << 16);
    }
    const uint32_t offset = ((lane & 15) * 32 + (lane >> 4) * 8 + tile * 16) * 2;
    const uint32_t address = destination + (offset ^ ((offset >> 3) & 0x30));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
                 :: "r"(address), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
  }
}


__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(THREADS)
void gemm(const __grid_constant__ CUtensorMap a_map,
          const __grid_constant__ CUtensorMap b_map,
          const __grid_constant__ CUtensorMap c_map, int reduction_iterations) {
  extern __shared__ __align__(1024) unsigned char shared[];
  const uint32_t base = uint32_t(__cvta_generic_to_shared(shared));
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  uint32_t rank;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  cluster_sync();
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(base + TMEM_ADDRESS), "r"(TMEM_COLUMNS) : "memory");
  }
  __syncthreads();
  if (warp == 0 && elect_one()) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&a_map));
    asm volatile("prefetch.tensormap [%0];" :: "l"(&b_map));
    asm volatile("prefetch.tensormap [%0];" :: "l"(&c_map));
#pragma unroll
    for (int stage = 0; stage < AB_STAGES; ++stage) {
      init_barrier(base + TMA_BARRIER + stage * 8, 1);
      init_barrier(base + MMA_BARRIER + stage * 8, 1);
    }
    init_barrier(base + COMPUTE_BARRIER, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cluster_sync();
  const uint32_t tmem = *reinterpret_cast<uint32_t*>(shared + TMEM_ADDRESS);
  constexpr uint32_t instruction = (1u << 4) | (1u << 7) | (1u << 10) |
                                   ((MMA_N / 8) << 17) | ((MMA_M / 16) << 24);
  if (warp == 4) {
    if (elect_one()) {
      Pipeline<AB_STAGES> producer{0, 1};
      for (int iteration = 0; iteration < reduction_iterations; ++iteration) {
        wait_barrier(base + MMA_BARRIER + producer.index * 8, producer.phase);
        if (rank == 0) {
          asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                       :: "r"(base + TMA_BARRIER + producer.index * 8),
                          "r"(2 * (A_BYTES + B_BYTES)) : "memory");
        }
        load_operand(base + producer.index * A_BYTES, &a_map, blockIdx.x * BLOCK_M,
                     iteration, base + TMA_BARRIER + producer.index * 8, rank);
        load_operand(base + B_OFFSET + producer.index * B_BYTES, &b_map,
                     blockIdx.y * MMA_N + rank * BLOCK_N, iteration,
                     base + TMA_BARRIER + producer.index * 8, rank);
        producer.step();
      }
    }
  } else if (warp == 5 && rank == 0) {
    Pipeline<AB_STAGES> consumer;
    for (int iteration = 0; iteration < reduction_iterations; ++iteration) {
      wait_barrier(base + TMA_BARRIER + consumer.index * 8, consumer.phase);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      if (elect_one()) {
#pragma unroll
        for (int step = 0; step < BLOCK_K / 16; ++step) {
          const uint64_t a_descriptor = descriptor(base + consumer.index * A_BYTES + step * 32);
          const uint64_t b_descriptor = descriptor(base + B_OFFSET + consumer.index * B_BYTES + step * 32);
          const uint32_t accumulate = iteration != 0 || step != 0;
          asm volatile(
              "{ .reg .pred accumulate; setp.ne.u32 accumulate, %4, 0; "
              "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, accumulate; }"
              :: "r"(tmem), "l"(a_descriptor), "l"(b_descriptor),
                 "r"(instruction), "r"(accumulate) : "memory");
        }
        commit_mma(base + MMA_BARRIER + consumer.index * 8);
      }
      consumer.step();
    }
    if (elect_one()) commit_mma(base + COMPUTE_BARRIER);
  } else if (warp < 4) {
    wait_barrier(base + COMPUTE_BARRIER, 0);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  #pragma unroll
    for (int stage = 0; stage < MMA_N / TMA_N; ++stage) {
      float upper[16], lower[16];
      const uint32_t address = tmem + stage * TMA_N + (uint32_t(warp * 32) << 16);
      load_fragment(address, upper);
      load_fragment(address + (16 << 16), lower);
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      const uint32_t destination = base + C_OFFSET + (stage % 2) * C_STAGE_BYTES;
      store_fragment(destination + warp * 32 * TMA_N * 2, upper, lane);
      store_fragment(destination + (warp * 32 + 16) * TMA_N * 2, lower, lane);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("bar.sync 1, 128;" ::: "memory");
      if (warp == 0 && lane == 0) {
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
            :: "l"(&c_map), "r"(int(blockIdx.y * MMA_N + stage * TMA_N)),
               "r"(int(blockIdx.x * BLOCK_M)), "r"(destination) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        if (stage == MMA_N / TMA_N - 1) {
          asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        } else {
          asm volatile("cp.async.bulk.wait_group 1;" ::: "memory");
        }
      }
      if (stage > 0 && stage < MMA_N / TMA_N - 1) {
        asm volatile("bar.sync 1, 128;" ::: "memory");
      }
    }

  }
  cluster_sync();
  if (warp == 0) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem), "r"(TMEM_COLUMNS) : "memory");
  }
  cluster_sync();
}

CUresult encode_tensor_map(CUtensorMap* tensor_map, __nv_bfloat16* pointer,
                           int rows, int reduction, int tile_rows) {
  const uint64_t dimensions[] = {uint64_t(reduction), uint64_t(rows)};
  const uint64_t strides[] = {uint64_t(reduction) * 2};
  const uint32_t box[] = {uint32_t(BLOCK_K), uint32_t(tile_rows)};
  const uint32_t element_strides[] = {1, 1};
  return cuTensorMapEncodeTiled(
      tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, pointer,
      dimensions, strides, box, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

CUresult encode_output(CUtensorMap* tensor_map, __nv_bfloat16* pointer, int rows, int columns) {
  const uint64_t dimensions[] = {uint64_t(columns), uint64_t(rows)};
  const uint64_t strides[] = {uint64_t(columns) * 2};
  const uint32_t box[] = {TMA_N, BLOCK_M};
  const uint32_t element_strides[] = {1, 1};
  return cuTensorMapEncodeTiled(
      tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, pointer,
      dimensions, strides, box, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void launch(const CUtensorMap& a_map, const CUtensorMap& b_map, const CUtensorMap& c_map,
            int rows, int columns, int reduction,
            cudaStream_t stream) {
  gemm<<<dim3(rows / BLOCK_M, columns / MMA_N), THREADS, SHARED_BYTES, stream>>>(
      a_map, b_map, c_map, reduction / BLOCK_K);
}

}
