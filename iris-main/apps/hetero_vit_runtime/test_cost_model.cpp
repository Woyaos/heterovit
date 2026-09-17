#include <cassert>
#include <cmath>
#include <cstdio>

#include "hetero_cost_model.h"

int main() {
  assert(HETERO_META_COUNT == 9);
  assert(HETERO_PLACEMENT_GPU != HETERO_PLACEMENT_FPGA);
  HeteroPolicyConfig config = {4000.0, 20.0, 50000.0, 2.0, 0.5, 0.25, 1.0, 1};
  double gpu = hetero_gpu_finish_us(&config, 1041, 0);
  double fpga_latency = hetero_fpga_finish_us(
      &config, 337, 302592, 302592, 1, 0, HETERO_OBJECTIVE_LATENCY);
  double fpga_throughput = hetero_fpga_finish_us(
      &config, 337, 302592, 302592, 1, 0, HETERO_OBJECTIVE_THROUGHPUT);
  double fpga_resident = hetero_fpga_finish_us(
      &config, 337, 302592, 302592, 0, 0, HETERO_OBJECTIVE_LATENCY);
  assert(fpga_latency < gpu);
  assert(fpga_throughput < fpga_latency);
  assert(fpga_resident < fpga_latency);
  assert(hetero_gpu_finish_us(&config, 1041, 2) > gpu);
  std::printf("gpu=%.3f us fpga_latency=%.3f us fpga_throughput=%.3f us\n",
              gpu, fpga_latency, fpga_throughput);
  return 0;
}
