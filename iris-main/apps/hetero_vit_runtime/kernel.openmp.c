#include <iris/iris_openmp.h>

#ifdef __cplusplus
extern "C" {
#endif

void hetero_noop(IRIS_OPENMP_KERNEL_ARGS) { (void)_off; (void)_bws; }

int iris_openmp_kernel(const char* name) {
  iris_openmp_lock();
  if (strcmp(name, "hetero_noop") == 0) {
    iris_openmp_kernel_idx = 0;
    return IRIS_SUCCESS;
  }
  iris_openmp_unlock();
  return IRIS_ERROR;
}

int iris_openmp_setarg(int idx, size_t size, void* value) {
  (void)idx; (void)size; (void)value;
  return IRIS_ERROR;
}

int iris_openmp_setmem(int idx, void* mem) {
  (void)idx; (void)mem;
  return IRIS_ERROR;
}

int iris_openmp_launch(int dim, size_t* off, size_t* ndr) {
  (void)dim;
  if (iris_openmp_kernel_idx == 0) hetero_noop(off, ndr);
  iris_openmp_unlock();
  return IRIS_SUCCESS;
}

#ifdef __cplusplus
}
#endif
