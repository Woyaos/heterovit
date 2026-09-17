#include <iris/iris.h>
#include <iris/rt/Device.h>
#include <iris/rt/Policy.h>
#include <iris/rt/Scheduler.h>
#include <iris/rt/Task.h>

#include <limits>

#include "hetero_cost_model.h"

namespace iris {
namespace rt {

class PolicyHeteroEFT : public Policy {
 public:
  PolicyHeteroEFT() : config_(NULL) {}
  virtual ~PolicyHeteroEFT() {}

  virtual void Init(void* params) { config_ = static_cast<HeteroPolicyConfig*>(params); }

  virtual void GetDevices(Task* task, Device** devs, int* ndevs) {
    Device* gpu = NULL;
    Device* fpga = NULL;
    Device* fallback = NULL;
    int gpu_index = -1;
    int fpga_index = -1;

    scheduler_->RefreshNTasksOnDevs();
    for (int i = 0; i < ndevices(); ++i) {
      Device* candidate = device(i);
      if (!IsKernelSupported(task, candidate)) continue;
      if (fallback == NULL) fallback = candidate;
      if (gpu == NULL && (candidate->type() & iris_nvidia)) {
        gpu = candidate;
        gpu_index = i;
      }
      if (fpga == NULL && (candidate->type() & iris_fpga)) {
        fpga = candidate;
        fpga_index = i;
      }
    }

    if (fallback == NULL) {
      *ndevs = 0;
      return;
    }
    if (config_ == NULL || task->n_metadata() < HETERO_META_COUNT) {
      devs[0] = gpu != NULL ? gpu : fallback;
      *ndevs = 1;
      return;
    }

    int* metadata = task->metadata();
    bool fpga_eligible = metadata[HETERO_META_FPGA_ELIGIBLE] != 0;
    int planned_device = metadata[HETERO_META_PLANNED_DEVICE];
    if (planned_device == HETERO_PLACEMENT_GPU) {
      if (gpu == NULL) { *ndevs = 0; return; }
      devs[0] = gpu;
      *ndevs = 1;
      return;
    }
    if (planned_device == HETERO_PLACEMENT_FPGA) {
      if (fpga == NULL || !fpga_eligible) { *ndevs = 0; return; }
      devs[0] = fpga;
      *ndevs = 1;
      return;
    }
    if (planned_device != HETERO_PLACEMENT_AUTO) { *ndevs = 0; return; }
    if (gpu == NULL || fpga == NULL || !fpga_eligible) {
      devs[0] = gpu != NULL ? gpu : fallback;
      *ndevs = 1;
      return;
    }

    int gpu_queue = static_cast<int>(scheduler_->NTasksOnDev(gpu_index));
    int fpga_queue = static_cast<int>(scheduler_->NTasksOnDev(fpga_index));
    double gpu_finish = hetero_gpu_finish_us(
        config_, metadata[HETERO_META_GPU_US], gpu_queue);
    double fpga_finish = hetero_fpga_finish_us(
        config_,
        metadata[HETERO_META_FPGA_US],
        metadata[HETERO_META_INPUT_BYTES],
        metadata[HETERO_META_OUTPUT_BYTES],
        metadata[HETERO_META_RETURN_REQUIRED],
        fpga_queue,
        metadata[HETERO_META_OBJECTIVE]);

    devs[0] = fpga_finish < gpu_finish ? fpga : gpu;
    *ndevs = 1;
  }

 private:
  HeteroPolicyConfig* config_;
};

}  // namespace rt
}  // namespace iris

REGISTER_CUSTOM_POLICY(PolicyHeteroEFT, hetero_eft)
