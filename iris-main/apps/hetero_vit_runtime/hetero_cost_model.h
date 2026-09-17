#ifndef IRIS_APPS_HETERO_VIT_COST_MODEL_H
#define IRIS_APPS_HETERO_VIT_COST_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

enum {
  HETERO_META_GPU_US = 0,
  HETERO_META_FPGA_US = 1,
  HETERO_META_INPUT_BYTES = 2,
  HETERO_META_OUTPUT_BYTES = 3,
  HETERO_META_FPGA_ELIGIBLE = 4,
  HETERO_META_OBJECTIVE = 5,
  HETERO_META_RETURN_REQUIRED = 6,
  HETERO_META_LINEAR_EQUIVALENTS = 7,
  HETERO_META_PLANNED_DEVICE = 8,
  HETERO_META_COUNT = 9
};

enum {
  HETERO_PLACEMENT_AUTO = 0,
  HETERO_PLACEMENT_GPU = 1,
  HETERO_PLACEMENT_FPGA = 2
};

enum {
  HETERO_OBJECTIVE_LATENCY = 0,
  HETERO_OBJECTIVE_THROUGHPUT = 1
};

typedef struct {
  double link_bandwidth_bytes_per_us;
  double one_way_fixed_latency_us;
  double codec_bandwidth_bytes_per_us;
  double codec_fixed_latency_us;
  double compression_ratio;
  double throughput_communication_weight;
  double queue_weight;
  int copies_per_direction;
} HeteroPolicyConfig;

static inline double hetero_one_way_transfer_us(
    const HeteroPolicyConfig* config, int encoded_bytes) {
  if (encoded_bytes <= 0) return 0.0;
  double transfer = config->copies_per_direction *
      (config->one_way_fixed_latency_us +
       encoded_bytes / config->link_bandwidth_bytes_per_us);
  double codec = config->codec_fixed_latency_us;
  if (config->codec_bandwidth_bytes_per_us > 0.0 && config->compression_ratio > 0.0) {
    double uncompressed_bytes = encoded_bytes / config->compression_ratio;
    codec += uncompressed_bytes / config->codec_bandwidth_bytes_per_us;
  }
  return transfer + 2.0 * codec;
}

static inline double hetero_gpu_finish_us(
    const HeteroPolicyConfig* config, int gpu_compute_us, int gpu_queue_depth) {
  return gpu_compute_us * (1.0 + config->queue_weight * gpu_queue_depth);
}

static inline double hetero_fpga_finish_us(
    const HeteroPolicyConfig* config,
    int fpga_compute_us,
    int input_bytes,
    int output_bytes,
    int return_required,
    int fpga_queue_depth,
    int objective) {
  double communication = hetero_one_way_transfer_us(config, input_bytes);
  if (return_required)
    communication += hetero_one_way_transfer_us(config, output_bytes);
  if (objective == HETERO_OBJECTIVE_THROUGHPUT)
    communication *= config->throughput_communication_weight;
  return fpga_compute_us * (1.0 + config->queue_weight * fpga_queue_depth) +
         communication;
}

#ifdef __cplusplus
}
#endif

#endif
