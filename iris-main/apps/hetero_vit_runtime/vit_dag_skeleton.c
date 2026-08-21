#include <iris/iris.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hetero_cost_model.h"
#include "vit_task_manifest.h"

int main(int argc, char** argv) {
  int objective = HETERO_OBJECTIVE_LATENCY;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--throughput") == 0) objective = HETERO_OBJECTIVE_THROUGHPUT;
    else if (strcmp(argv[i], "--latency") == 0) objective = HETERO_OBJECTIVE_LATENCY;
    else {
      fprintf(stderr, "usage: %s [--latency|--throughput]\n", argv[0]);
      return 2;
    }
  }

  int error = iris_init(&argc, &argv, 1);
  if (error != IRIS_SUCCESS) return error;

  HeteroPolicyConfig config = {
      4000.0,  // 4 GB/s = 4000 bytes/us
      20.0,
      50000.0,
      2.0,
      0.5,
      0.25,
      1.0,
      1,
  };
  error = iris_register_policy("libPolicyHeteroEFT.so", "hetero_eft", &config);
  if (error != IRIS_SUCCESS) {
    fprintf(stderr, "failed to register hetero_eft policy\n");
    iris_finalize();
    return error;
  }

  iris_task* tasks = (iris_task*)calloc(HETERO_TASK_COUNT, sizeof(iris_task));
  int* dependency_count = (int*)calloc(HETERO_TASK_COUNT, sizeof(int));
  if (tasks == NULL || dependency_count == NULL) {
    free(dependency_count);
    free(tasks);
    iris_finalize();
    return 2;
  }

  size_t one = 1;
  for (int i = 0; i < HETERO_TASK_COUNT; ++i) {
    iris_task_create(&tasks[i]);
    iris_task_set_name(tasks[i], HETERO_TASKS[i].name);
    iris_task_set_metadata_all(tasks[i], (int*)HETERO_TASKS[i].metadata, HETERO_META_COUNT);
    iris_task_set_metadata(tasks[i], HETERO_META_OBJECTIVE, objective);
    iris_task_kernel(tasks[i], "hetero_noop", 1, NULL, &one, NULL, 0, NULL, NULL);
  }

  for (int i = 0; i < HETERO_EDGE_COUNT; ++i)
    dependency_count[HETERO_EDGES[i].target]++;
  for (int target = 0; target < HETERO_TASK_COUNT; ++target) {
    if (dependency_count[target] == 0) continue;
    iris_task* dependencies =
        (iris_task*)malloc(dependency_count[target] * sizeof(iris_task));
    int cursor = 0;
    for (int edge = 0; edge < HETERO_EDGE_COUNT; ++edge) {
      if (HETERO_EDGES[edge].target == target)
        dependencies[cursor++] = tasks[HETERO_EDGES[edge].source];
    }
    iris_task_depend(tasks[target], dependency_count[target], dependencies);
    free(dependencies);
  }

  for (int i = 0; i < HETERO_TASK_COUNT; ++i)
    iris_task_submit(tasks[i], iris_custom, "hetero_eft", 0);
  iris_task_wait(tasks[HETERO_TASK_COUNT - 1]);

  printf("submitted %d ViT tasks and %d dependencies through hetero_eft (%s objective)\n",
         HETERO_TASK_COUNT, HETERO_EDGE_COUNT,
         objective == HETERO_OBJECTIVE_THROUGHPUT ? "throughput" : "latency");
  for (int i = 0; i < HETERO_TASK_COUNT; ++i) iris_task_release(tasks[i]);
  free(dependency_count);
  free(tasks);
  iris_finalize();
  return iris_error_count();
}
