// Experiment harness for mojo_v8: isolates mainloop / TMA feed / epilogue cost.
// Build variants: (none)=base, SKIP_EPILOGUE, SKIP_MMA, INSTRUMENT,
// BIGBOX2 (one TMA op per 2 K-stages), L2_PROMO, NOMULTICAST.
// -DAB_STAGES=n and -DC_BUFFERS=n override pipeline depths. Not a correctness test.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace mojo_v8 {

constexpr int BLOCK_M = 128;
#ifndef BLOCK_N
#define BLOCK_N 64
#endif
constexpr int BLOCK_K = 64;
constexpr int MMA_M = 256;
#ifndef MMA_N
#define MMA_N 128
#endif
#ifndef BIGBOX_N
#define BIGBOX_N 1
#endif
constexpr int THREADS = 224;
#ifndef AB_STAGES
#define AB_STAGES 8
#endif
#ifndef C_BUFFERS
#define C_BUFFERS 2
#endif
constexpr int ACCUM_STAGES = 512 / MMA_N;
constexpr int CLC_STAGES = 2;
constexpr int A_BYTES = BLOCK_M * BLOCK_K * 2;
constexpr int B_BYTES = BLOCK_N * BLOCK_K * 2;
constexpr int C_BYTES = BLOCK_M * 32 * 2;
constexpr int B_OFFSET = AB_STAGES * A_BYTES;
constexpr int C_OFFSET = B_OFFSET + AB_STAGES * B_BYTES;
constexpr int TMA_FULL = C_OFFSET + C_BUFFERS * C_BYTES;
constexpr int MMA_EMPTY = TMA_FULL + AB_STAGES * 8;
constexpr int ACCUM_FULL = MMA_EMPTY + AB_STAGES * 8;
constexpr int ACCUM_EMPTY = ACCUM_FULL + ACCUM_STAGES * 8;
constexpr int CLC_FULL = ACCUM_EMPTY + ACCUM_STAGES * 8;
constexpr int CLC_EMPTY = CLC_FULL + CLC_STAGES * 8;
constexpr int THROTTLE_FULL = CLC_EMPTY + CLC_STAGES * 8;
constexpr int THROTTLE_EMPTY = THROTTLE_FULL + CLC_STAGES * 8;
constexpr int CLC_RESPONSE = THROTTLE_EMPTY + CLC_STAGES * 8;
constexpr int DEALLOC_BARRIER = CLC_RESPONSE + CLC_STAGES * 16;
constexpr int TMEM_ADDRESS = DEALLOC_BARRIER + 8;
constexpr int SHARED_BYTES = TMEM_ADDRESS + 4;
static_assert(CLC_RESPONSE % 16 == 0);

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

__device__ __forceinline__ uint32_t remote_address(uint32_t address, uint32_t rank) {
  uint32_t result;
  asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(result) : "r"(address), "r"(rank));
  return result;
}

__device__ __forceinline__ void init_barrier(uint32_t address, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(address), "r"(count) : "memory");
}

__device__ __forceinline__ void wait_barrier(uint32_t address, uint32_t phase) {
  asm volatile(
      "{ .reg .pred ready; wait_loop: "
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%0], %1; "
      "@!ready bra wait_loop; }"
      :: "r"(address), "r"(phase) : "memory");
}

__device__ __forceinline__ void arrive(uint32_t address, uint32_t rank) {
  const uint32_t destination = remote_address(address, rank);
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
               :: "r"(destination) : "memory");
}

__device__ __forceinline__ void expect_bytes(uint32_t address, uint32_t rank, int bytes) {
  const uint32_t destination = remote_address(address, rank);
  asm volatile("mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [%0], %1;"
               :: "r"(destination), "r"(bytes) : "memory");
}

__device__ __forceinline__ void commit_mma(uint32_t address) {
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
      :: "r"(address), "h"(uint16_t(3)) : "memory");
}

__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ void epilogue_sync() {
  asm volatile("bar.sync 2, 128;" ::: "memory");
}

#ifdef INSTRUMENT
__device__ unsigned long long g_stats[8];
// slots: 0=mma TMA_FULL wait, 1=mma ACCUM_EMPTY wait, 2=producer MMA_EMPTY wait,
//        3=epilogue ACCUM_FULL wait, 4=mma iterations, 5=mma total cycles,
//        6=producer iterations, 7=epilogue tiles
__device__ __forceinline__ bool instrumented(int lane, uint32_t rank) {
  return lane == 0 && blockIdx.x == 0 && blockIdx.y == 0;
}
#define TICK() unsigned long long tick__ = clock64()
#define TOCK(slot) do { if (instrumented(lane, rank)) atomicAdd(&g_stats[slot], clock64() - tick__); } while (0)
#define COUNT(slot) do { if (instrumented(lane, rank)) atomicAdd(&g_stats[slot], 1); } while (0)
#else
#define TICK()
#define TOCK(slot)
#define COUNT(slot)
#endif

struct Work {
  uint32_t row;
  uint32_t column;
  bool valid;
};

__device__ __forceinline__ Work map_work(uint32_t row, uint32_t column, bool valid, uint32_t rank) {
  uint32_t cluster_row = row / 2;
  if (column % 2) cluster_row = gridDim.x / 2 - cluster_row - 1;
  return {cluster_row * 2 + rank, column, valid};
}

__device__ __forceinline__ Work fetch_work(uint32_t base, Pipeline<CLC_STAGES>& state, uint32_t rank) {
  wait_barrier(base + CLC_FULL + state.index * 8, state.phase);
  uint32_t row = 0, column = 0, depth = 0, valid = 0;
  const uint32_t response = base + CLC_RESPONSE + state.index * 16;
  asm volatile(
      "{ .reg .pred canceled; .reg .b128 result; "
      "ld.shared.b128 result, [%4]; "
      "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 canceled, result; "
      "@canceled clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, result; }"
      : "+r"(row), "+r"(column), "+r"(depth), "+r"(valid) : "r"(response) : "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  arrive(base + CLC_EMPTY + state.index * 8, 0);
  state.step();
  return map_work(row, column, valid != 0, rank);
}

__device__ __forceinline__ void load_operand(uint32_t destination, const CUtensorMap* tensor_map,
                                            int row, int reduction_tile, uint32_t barrier, uint32_t rank) {
  const uint32_t leader_barrier = remote_address(barrier, 0);
#ifdef NOMULTICAST
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 "
      "[%0], [%1, {0, %2, %3}], [%4];"
      :: "r"(destination), "l"(tensor_map), "r"(row), "r"(reduction_tile),
         "r"(leader_barrier) : "memory");
#else
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2 "
      "[%0], [%1, {0, %2, %3}], [%4], %5;"
      :: "r"(destination), "l"(tensor_map), "r"(row), "r"(reduction_tile),
         "r"(leader_barrier), "h"(uint16_t(1u << rank)) : "memory");
#endif
}

#if BIGBOX_N > 1
// One TMA op covers two consecutive K stages (box dim2 = 2).
__device__ __forceinline__ void load_operand2(uint32_t destination, const CUtensorMap* tensor_map,
                                              int row, int reduction_tile, uint32_t barrier, uint32_t rank) {
  const uint32_t leader_barrier = remote_address(barrier, 0);
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2 "
      "[%0], [%1, {0, %2, %3}], [%4], %5;"
      :: "r"(destination), "l"(tensor_map), "r"(row), "r"(reduction_tile),
         "r"(leader_barrier), "h"(uint16_t(1u << rank)) : "memory");
}
#endif

__device__ __forceinline__ uint64_t descriptor(uint32_t address) {
  return uint64_t(address >> 4) | (uint64_t(64) << 32) |
         (uint64_t(1) << 46) | (uint64_t(2) << 61);
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

__device__ __forceinline__ void store_output(uint32_t base, uint32_t tmem, const CUtensorMap* c_map,
                                            Work work, Pipeline<ACCUM_STAGES>& accum, int warp, int lane,
                                            uint32_t rank) {
  TICK();
  wait_barrier(base + ACCUM_FULL + accum.index * 8, accum.phase);
  TOCK(3);
  COUNT(7);
#ifdef SKIP_EPILOGUE
  arrive(base + ACCUM_EMPTY + accum.index * 8, 0);
  accum.step();
  return;
#else
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t tile_tmem = tmem + accum.index * MMA_N + ((rank * 128 + warp * 32) << 16);
#pragma unroll
  for (int stage = 0; stage < MMA_N / 32; ++stage) {
    float upper[16], lower[16];
    load_fragment(tile_tmem + stage * 32, upper);
    load_fragment(tile_tmem + stage * 32 + (16 << 16), lower);
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    if (stage == MMA_N / 32 - 1) arrive(base + ACCUM_EMPTY + accum.index * 8, 0);
    const uint32_t c_shared = base + C_OFFSET + (stage % C_BUFFERS) * C_BYTES;
    store_fragment(c_shared + warp * 32 * 64, upper, lane);
    store_fragment(c_shared + (warp * 32 + 16) * 64, lower, lane);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    epilogue_sync();
    if (warp == 0 && lane == 0) {
      asm volatile(
          "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
          :: "l"(c_map), "r"(int(work.column * MMA_N + stage * 32)),
             "r"(int(work.row * BLOCK_M)), "r"(c_shared) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      if (stage == MMA_N / 32 - 1) {
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
      } else {
        asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
      }
    }
    if (stage > 0) epilogue_sync();
  }
  accum.step();
#endif
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
  if (threadIdx.x == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&a_map));
    asm volatile("prefetch.tensormap [%0];" :: "l"(&b_map));
    asm volatile("prefetch.tensormap [%0];" :: "l"(&c_map));
    for (int stage = 0; stage < AB_STAGES; ++stage) {
      init_barrier(base + TMA_FULL + stage * 8, 1);
      init_barrier(base + MMA_EMPTY + stage * 8, 1);
    }
    for (int stage = 0; stage < ACCUM_STAGES; ++stage) {
      init_barrier(base + ACCUM_FULL + stage * 8, 1);
      init_barrier(base + ACCUM_EMPTY + stage * 8, 256);
    }
    for (int stage = 0; stage < CLC_STAGES; ++stage) {
      init_barrier(base + CLC_FULL + stage * 8, 1);
      init_barrier(base + CLC_EMPTY + stage * 8, 416);
      init_barrier(base + THROTTLE_FULL + stage * 8, 32);
      init_barrier(base + THROTTLE_EMPTY + stage * 8, 32);
    }
    init_barrier(base + DEALLOC_BARRIER, 256);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cluster_sync();

  Work work = map_work(blockIdx.x, blockIdx.y, true, rank);
  Pipeline<CLC_STAGES> clc_consumer;

  if (warp == 5) {
    Pipeline<AB_STAGES> producer{0, 1};
    Pipeline<CLC_STAGES> throttle{0, 1};
    while (work.valid) {
      if (rank == 0) {
        wait_barrier(base + THROTTLE_EMPTY + throttle.index * 8, throttle.phase);
        arrive(base + THROTTLE_FULL + throttle.index * 8, 0);
        throttle.step();
      }
#if BIGBOX_N > 1
      for (int iteration = 0; iteration < reduction_iterations; ++iteration) {
        TICK();
        wait_barrier(base + MMA_EMPTY + producer.index * 8, producer.phase);
        TOCK(2);
        COUNT(6);
        if (lane == 0 && (producer.index % BIGBOX_N == 0)) {
          if (rank == 0) expect_bytes(base + TMA_FULL + producer.index * 8, 0, 2 * BIGBOX_N * (A_BYTES + B_BYTES));
          load_operand2(base + producer.index * A_BYTES, &a_map, work.row * BLOCK_M,
                        iteration, base + TMA_FULL + producer.index * 8, rank);
          load_operand2(base + B_OFFSET + producer.index * B_BYTES, &b_map,
                        work.column * MMA_N + rank * BLOCK_N, iteration,
                        base + TMA_FULL + producer.index * 8, rank);
        }
        producer.step();
      }
#else
      for (int iteration = 0; iteration < reduction_iterations; ++iteration) {
        TICK();
        wait_barrier(base + MMA_EMPTY + producer.index * 8, producer.phase);
        TOCK(2);
        COUNT(6);
        if (lane == 0) {
          if (rank == 0) expect_bytes(base + TMA_FULL + producer.index * 8, 0, 2 * (A_BYTES + B_BYTES));
          load_operand(base + producer.index * A_BYTES, &a_map, work.row * BLOCK_M,
                       iteration, base + TMA_FULL + producer.index * 8, rank);
          load_operand(base + B_OFFSET + producer.index * B_BYTES, &b_map,
                       work.column * MMA_N + rank * BLOCK_N, iteration,
                       base + TMA_FULL + producer.index * 8, rank);
        }
        producer.step();
      }
#endif
      __syncwarp();
      work = fetch_work(base, clc_consumer, rank);
    }
    for (int stage = 0; stage < AB_STAGES; ++stage) {
      wait_barrier(base + MMA_EMPTY + producer.index * 8, producer.phase);
      producer.step();
    }
  } else if (warp == 4 && rank == 0) {
    Pipeline<CLC_STAGES> clc_producer{0, 1};
    Pipeline<CLC_STAGES> throttle;
    while (work.valid) {
      wait_barrier(base + THROTTLE_FULL + throttle.index * 8, throttle.phase);
      arrive(base + THROTTLE_EMPTY + throttle.index * 8, 0);
      throttle.step();
      wait_barrier(base + CLC_EMPTY + clc_producer.index * 8, clc_producer.phase);
      if (lane < 2) expect_bytes(base + CLC_FULL + clc_producer.index * 8, lane, 16);
      __syncwarp();
      if (lane == 0) {
        asm volatile(
            "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];"
            :: "r"(base + CLC_RESPONSE + clc_producer.index * 16),
               "r"(base + CLC_FULL + clc_producer.index * 8) : "memory");
      }
      clc_producer.step();
      work = fetch_work(base, clc_consumer, rank);
    }
    for (int stage = 0; stage < CLC_STAGES; ++stage) {
      wait_barrier(base + CLC_EMPTY + clc_producer.index * 8, clc_producer.phase);
      clc_producer.step();
    }
  } else if (warp == 6) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;"
                 :: "r"(base + TMEM_ADDRESS) : "memory");
    __syncwarp();
    asm volatile("bar.arrive 1, 160;" ::: "memory");
    const uint32_t tmem = *reinterpret_cast<uint32_t*>(shared + TMEM_ADDRESS);
    unsigned long long mma_start = clock64();
    Pipeline<AB_STAGES> consumer;
    Pipeline<ACCUM_STAGES> accum{0, 1};
    constexpr uint32_t instruction = (1u << 4) | (1u << 7) | (1u << 10) |
                                     ((MMA_N / 8) << 17) | ((MMA_M / 16) << 24);
    while (work.valid) {
      Work next = fetch_work(base, clc_consumer, rank);
      if (rank == 0) {
        TICK();
        wait_barrier(base + ACCUM_EMPTY + accum.index * 8, accum.phase);
        TOCK(1);
        const uint32_t destination = tmem + accum.index * MMA_N;
        for (int iteration = 0; iteration < reduction_iterations; ++iteration) {
          TICK();
#if BIGBOX_N > 1
          if (consumer.index % BIGBOX_N == 0)
#endif
          {
            wait_barrier(base + TMA_FULL + consumer.index * 8, consumer.phase);
          }
          TOCK(0);
          COUNT(4);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          if (lane == 0) {
#ifdef SKIP_MMA
            arrive(base + MMA_EMPTY + consumer.index * 8, 0);
            arrive(base + MMA_EMPTY + consumer.index * 8, 1);
#else
#pragma unroll
            for (int step = 0; step < 4; ++step) {
              const uint64_t a_descriptor = descriptor(base + consumer.index * A_BYTES + step * 32);
              const uint64_t b_descriptor = descriptor(base + B_OFFSET + consumer.index * B_BYTES + step * 32);
              const uint32_t accumulate = iteration != 0 || step != 0;
              asm volatile(
                  "{ .reg .pred accumulate; setp.ne.u32 accumulate, %4, 0; "
                  "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, accumulate; }"
                  :: "r"(destination), "l"(a_descriptor), "l"(b_descriptor),
                     "r"(instruction), "r"(accumulate) : "memory");
            }
            commit_mma(base + MMA_EMPTY + consumer.index * 8);
#endif
          }
          consumer.step();
        }
        if (lane == 0) {
#ifdef SKIP_MMA
          arrive(base + ACCUM_FULL + accum.index * 8, 0);
          arrive(base + ACCUM_FULL + accum.index * 8, 1);
#else
          commit_mma(base + ACCUM_FULL + accum.index * 8);
#endif
        }
        accum.step();
      }
      work = next;
    }
#ifdef INSTRUMENT
    if (instrumented(lane, rank)) atomicAdd(&g_stats[5], clock64() - mma_start);
#endif
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
    wait_barrier(base + DEALLOC_BARRIER, 0);
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" :: "r"(tmem) : "memory");
  } else if (warp < 4) {
    asm volatile("bar.sync 1, 160;" ::: "memory");
    const uint32_t tmem = *reinterpret_cast<uint32_t*>(shared + TMEM_ADDRESS);
    Pipeline<ACCUM_STAGES> accum;
    while (work.valid) {
      store_output(base, tmem, &c_map, work, accum, warp, lane, rank);
      work = fetch_work(base, clc_consumer, rank);
    }
    arrive(base + DEALLOC_BARRIER, rank ^ 1);
    arrive(base + DEALLOC_BARRIER, rank);
  }
  cluster_sync();
}

CUresult encode_operand(CUtensorMap* tensor_map, __nv_bfloat16* pointer,
                        int rows, int reduction, int tile_rows) {
  const uint64_t dimensions[] = {64, uint64_t(rows), uint64_t(reduction / 64)};
  const uint64_t strides[] = {uint64_t(reduction) * 2, 128};
#if BIGBOX_N > 1
  const uint32_t box[] = {64, uint32_t(tile_rows), BIGBOX_N};
#else
  const uint32_t box[] = {64, uint32_t(tile_rows), 1};
#endif
  const uint32_t element_strides[] = {1, 1, 1};
#ifdef L2_PROMO
  const CUtensorMapL2promotion promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
#else
  const CUtensorMapL2promotion promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
#endif
  return cuTensorMapEncodeTiled(tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, pointer,
      dimensions, strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, promotion, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

CUresult encode_output(CUtensorMap* tensor_map, __nv_bfloat16* pointer, int rows, int columns) {
  const uint64_t dimensions[] = {uint64_t(columns), uint64_t(rows)};
  const uint64_t strides[] = {uint64_t(columns) * 2};
  const uint32_t box[] = {32, BLOCK_M};
  const uint32_t element_strides[] = {1, 1};
  return cuTensorMapEncodeTiled(tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, pointer,
      dimensions, strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void launch(const CUtensorMap& a_map, const CUtensorMap& b_map, const CUtensorMap& c_map,
            int rows, int columns, int reduction, cudaStream_t stream) {
  gemm<<<dim3(rows / BLOCK_M, columns / MMA_N), THREADS, SHARED_BYTES, stream>>>(
      a_map, b_map, c_map, reduction / BLOCK_K);
}

}

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(status));
    std::exit(1);
  }
}
void check(CUresult status, const char* what) {
  if (status != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(status, &message);
    std::fprintf(stderr, "Driver error at %s: %s\n", what, message ? message : "?");
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "Usage: exp M N K [fill_byte_hex]\n");
    return 1;
  }
  const int m = std::atoi(argv[1]), n = std::atoi(argv[2]), k = std::atoi(argv[3]);
  const int fill = argc > 4 ? std::stoi(argv[4], nullptr, 16) : 0;
#if defined(SKIP_EPILOGUE)
  const char* variant = "noepi";
#elif defined(SKIP_MMA)
  const char* variant = "nomma";
#else
  static char variant_storage[64];
  std::snprintf(variant_storage, sizeof(variant_storage), "bn%d_mman%d_bb%d_st%d%s%s", BLOCK_N,
                MMA_N, BIGBOX_N, AB_STAGES,
#ifdef L2_PROMO
                "_promo",
#else
                "",
#endif
#ifdef NOMULTICAST
                "_nomcast"
#else
                ""
#endif
  );
  const char* variant = variant_storage;
#endif
  cudaStream_t stream;
  check(cudaStreamCreate(&stream), "stream");
  __nv_bfloat16 *a, *b, *c;
  check(cudaMalloc(&a, size_t(m) * k * 2), "a");
  check(cudaMalloc(&b, size_t(n) * k * 2), "b");
  check(cudaMalloc(&c, size_t(m) * n * 2), "c");
  check(cudaMemsetAsync(a, fill, size_t(m) * k * 2, stream), "memset a");
  check(cudaMemsetAsync(b, fill, size_t(n) * k * 2, stream), "memset b");
  alignas(64) CUtensorMap a_map, b_map, c_map;
  check(mojo_v8::encode_operand(&a_map, a, m, k, mojo_v8::BLOCK_M), "a_map");
  check(mojo_v8::encode_operand(&b_map, b, n, k, mojo_v8::BLOCK_N), "b_map");
  check(mojo_v8::encode_output(&c_map, c, m, n), "c_map");
  check(cudaFuncSetAttribute(mojo_v8::gemm, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             mojo_v8::SHARED_BYTES), "smem attr");
  const int warmup = 20, iterations = 200;
  for (int i = 0; i < warmup; ++i) mojo_v8::launch(a_map, b_map, c_map, m, n, k, stream);
  check(cudaStreamSynchronize(stream), "warmup");
  check(cudaGetLastError(), "launch");
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "ev");
  check(cudaEventCreate(&stop), "ev");
  check(cudaEventRecord(start, stream), "rec");
  for (int i = 0; i < iterations; ++i) mojo_v8::launch(a_map, b_map, c_map, m, n, k, stream);
  check(cudaEventRecord(stop, stream), "rec");
  check(cudaEventSynchronize(stop), "sync");
  float ms = 0;
  check(cudaEventElapsedTime(&ms, start, stop), "time");
  const double per_launch = ms / iterations;
  const double tflops = 2.0 * m * n * k / (per_launch * 1e-3) / 1e12;
  std::printf("EXP variant=%s M=%d N=%d K=%d: %.6f ms, %.2f TFLOPS (stream loop)\n",
              variant, m, n, k, per_launch, tflops);

  // Same measurement via a captured CUDA graph of 100 launches.
  cudaGraph_t graph;
  cudaGraphExec_t executable;
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture begin");
  for (int i = 0; i < 100; ++i) mojo_v8::launch(a_map, b_map, c_map, m, n, k, stream);
  check(cudaStreamEndCapture(stream, &graph), "capture end");
  check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate");
  check(cudaGraphDestroy(graph), "graph destroy");
  check(cudaGraphLaunch(executable, stream), "graph warmup");
  check(cudaStreamSynchronize(stream), "graph warmup sync");
  check(cudaEventRecord(start, stream), "rec");
  check(cudaGraphLaunch(executable, stream), "graph launch");
  check(cudaEventRecord(stop, stream), "rec");
  check(cudaEventSynchronize(stop), "sync");
  check(cudaEventElapsedTime(&ms, start, stop), "time");
  const double graph_ms = ms / 100;
  std::printf("EXP variant=%s M=%d N=%d K=%d: %.6f ms, %.2f TFLOPS (graph replay)\n",
              variant, m, n, k, graph_ms, 2.0 * m * n * k / (graph_ms * 1e-3) / 1e12);
#ifdef INSTRUMENT
  unsigned long long stats[8];
  {
    void* dev = nullptr;
    check(cudaGetSymbolAddress(&dev, mojo_v8::g_stats), "sym");
    check(cudaMemcpy(stats, dev, sizeof(stats), cudaMemcpyDeviceToHost), "stats copy");
  }
  const double iters = std::max(1.0, double(stats[4]));
  std::printf(
      "STATS mma_tma_wait=%.1f mma_accum_wait=%.1f producer_mma_empty_wait=%.1f "
      "epi_accum_full_wait=%.1f cycles/iter; mma_iters=%llu producer_iters=%llu epi_tiles=%llu "
      "mma_total_cycles=%llu\n",
      stats[0] / iters, stats[1] / std::max(1.0, stats[7] ? double(stats[7]) : 1.0),
      stats[2] / std::max(1.0, double(stats[6])), stats[3] / std::max(1.0, double(stats[7])),
      stats[4], stats[6], stats[7], stats[5]);
#endif
  return 0;
}
