#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace mojo_v5 {

constexpr int BLOCK_M = 128;
constexpr int BLOCK_K = 64;
constexpr int MMA_M = 256;
constexpr int THREADS = 192;
constexpr int TMEM_COLUMNS = 512;
constexpr int BLOCK_N = 128;
constexpr int MMA_N = 256;
constexpr int TMA_N = 64;
constexpr int C_BYTES = BLOCK_M * MMA_N * 2;

constexpr int A_BYTES = BLOCK_M * BLOCK_K * 2;
constexpr int B_BYTES = BLOCK_N * BLOCK_K * 2;
constexpr int AB_STAGES = (233472 - C_BYTES) / (A_BYTES + B_BYTES + 32);
constexpr int B_OFFSET = AB_STAGES * A_BYTES;
constexpr int C_OFFSET = B_OFFSET + AB_STAGES * B_BYTES;
constexpr int TMA_BARRIER = C_OFFSET + C_BYTES;
constexpr int MMA_BARRIER = TMA_BARRIER + AB_STAGES * 8;
constexpr int COMPUTE_BARRIER = MMA_BARRIER + AB_STAGES * 8;
constexpr int TMEM_ADDRESS = COMPUTE_BARRIER + 8;
constexpr int SHARED_BYTES = C_OFFSET + C_BYTES + AB_STAGES * 32;

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

__device__ __forceinline__ void load_fragment(uint32_t read_address, float (&fragment)[128]) {
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
}

__device__ __forceinline__ void store_fragment(uint32_t destination,
                                              const float (&fragment)[128], int index) {
  uint32_t packed[4];
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const __nv_bfloat162 value = __float22bfloat162_rn(
        make_float2(fragment[index + pair * 2], fragment[index + pair * 2 + 1]));
    const __nv_bfloat162_raw raw = value;
    packed[pair] = uint32_t(raw.x) | (uint32_t(raw.y) << 16);
  }
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
               :: "r"(destination), "r"(packed[0]), "r"(packed[1]),
                  "r"(packed[2]), "r"(packed[3]) : "memory");
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
        // rank0/1 的 warp0 都要搬数据?
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
    float upper[128], lower[128];
    load_fragment(tmem | (uint32_t(warp * 32) << 16), upper);
    load_fragment(tmem | (uint32_t(warp * 32 + 16) << 16), lower);
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  #pragma unroll
    for (int output_tile = 0; output_tile < MMA_N / TMA_N; ++output_tile) {
  #pragma unroll
      for (int matrix = 0; matrix < TMA_N / 16; ++matrix) {
        const int index = output_tile * (TMA_N / 16) * 8 + matrix * 8;
        const uint32_t offset = ((lane & 15) * TMA_N + (lane >> 4) * 8 + matrix * 16) * 2;
        const uint32_t destination = base + C_OFFSET +
            (output_tile * BLOCK_M + warp * 32) * TMA_N * 2 +
            (offset ^ ((offset >> 3) & 0x70));
        store_fragment(destination, upper, index);
        store_fragment(destination + 16 * TMA_N * 2, lower, index);
      }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("bar.sync 1, 128;" ::: "memory");
    if (threadIdx.x < MMA_N / TMA_N) {
      const uint32_t source = base + C_OFFSET + threadIdx.x * BLOCK_M * TMA_N * 2;
      asm volatile(
          "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
          :: "l"(&c_map), "r"(int(blockIdx.y * MMA_N + threadIdx.x * TMA_N)),
             "r"(int(blockIdx.x * BLOCK_M)), "r"(source) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
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
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void launch(const CUtensorMap& a_map, const CUtensorMap& b_map, const CUtensorMap& c_map,
            int rows, int columns, int reduction,
            cudaStream_t stream) {
  gemm<<<dim3(rows / BLOCK_M, columns / MMA_N), THREADS, SHARED_BYTES, stream>>>(
      a_map, b_map, c_map, reduction / BLOCK_K);
}

}
