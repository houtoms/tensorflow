#include <cuda.h>
#include <stdio.h>

#define CK(cmd) do { \
  CUresult cmd_res = (cmd); \
  printf("%s returned %d", #cmd, cmd_res); \
  const char* result_name = NULL; \
  CUresult name_res = cuGetErrorName(cmd_res, &result_name); \
  if (name_res == CUDA_SUCCESS) { \
    printf(" (%s)\n", result_name); \
  } else { \
    printf("\n"); \
  } \
  if (cmd_res != CUDA_SUCCESS) { \
    printf("ABORTING\n"); \
    exit (1); \
  } \
} while (false)

int main(int argc, char* arg[]) {
  CUresult result;
  CUcontext ctx;
  CUdevice dev0 = 0;

  CK(cuInit(0));
  CK(cuDevicePrimaryCtxRetain(&ctx, dev0));
  CK(cuDevicePrimaryCtxRelease(dev0));
  exit (0);
}

