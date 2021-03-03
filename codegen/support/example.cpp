#include "generated_subgraph_0.h"

/**
 * Full description of one input/output buffer
 */
struct alignas(void*) HalideBuffer
{
  halide_dimension_t dims[4]; // 4 is current max number of dimensions supported by Halide
  halide_buffer_t buffer_data;
};

/**
 * Pointer to Halide generated function
 */
typedef int (*CompiledFuncImpl)(void **);

/**
 * Structure that stores data needed to execute Halide generated code wrapper
 */
struct alignas(void*) HalideConfiguration
{
  int num_arguments;
  CompiledFuncImpl impl;
  void **args;
  HalideBuffer *buffers;
};

/**
 * Pointer to wrapper function around generated code
 * First argument: pointer to configuration
 * Second argument: arrays of pointers to input and output buffers. Inputs goes first
 */
typedef int (*CompiledFuncWrapper)(char *, void **);

/**
 * Configured function, detached from underlying generator technology, like Halide.
 * All technology specific infomation should be stored in configuration array.
 */
struct ConfiguredCompiledFunc
{
  CompiledFuncWrapper wrapper;
  char *configuration;
};

static int halide_func_wrapper(char *configuration, void **args)
{
  auto *h_config = reinterpret_cast<HalideConfiguration *>(configuration);
  for (int i = 0; i < h_config->num_arguments; ++i)
  {
    h_config->buffers[i].buffer_data.host = static_cast<uint8_t *>(args[i]);
  }
  return h_config->impl(h_config->args);
}

static inline ConfiguredCompiledFunc create_generated_subgraph_impl(int ranks[], int *dims[], const halide_filter_metadata_t *metadata, CompiledFuncImpl func_impl)
{
  ConfiguredCompiledFunc func;
  func.wrapper = halide_func_wrapper;

/*
  Allocate and initilize memory for Halide structures.
  First place HalideConfiguration structure, then array of HalideBuffers, then array of void* that will be passed to generated function.
*/
  const int arguments = metadata->num_arguments;

  int need_memory = sizeof(HalideConfiguration) + arguments * (sizeof(HalideBuffer) + arguments*sizeof(void*));
  char *raw_config = func.configuration = new char[need_memory];
  
  auto *header = reinterpret_cast<HalideConfiguration *>(raw_config);
  auto *buffers = reinterpret_cast<HalideBuffer *>(raw_config + sizeof(HalideConfiguration));
  auto *args = reinterpret_cast<void **>(raw_config + sizeof(HalideConfiguration) + arguments * sizeof(HalideBuffer));

  header->num_arguments = arguments;
  header->impl = func_impl;
  header->args = args;
  header->buffers = buffers;

  for (int i = 0; i < arguments; ++i)
  {
    halide_buffer_t &halide_buf = buffers[i].buffer_data;
    halide_dimension_t *halide_dims = buffers[i].dims;

    args[i] = &halide_buf;

    halide_buf.device = 0;
    halide_buf.device_interface = nullptr;
    halide_buf.host = nullptr; // this attribute will be initialized during operator execution
    halide_buf.flags = 0;
    halide_buf.type = metadata->arguments[i].type;
    halide_buf.dimensions = ranks[i];
    halide_buf.dim = halide_dims;
    halide_buf.padding = nullptr;
    int stride = 1;
    for (int j = 0; j < ranks[i]; ++j)
    {
      halide_dimension_t &dim = halide_dims[j];
      dim.min = 0;
      dim.extent = dims[i][ranks[i] - 1 - j]; // flip indexes intentionally
      dim.stride = stride;
      stride *= dim.extent;
    }
  }
  return func;
}

// public interface

#define GENERATED_OPERATOR(name)\
ConfiguredCompiledFunc create_##name(int ranks[], int *dims[])\
{\
  const halide_filter_metadata_t *metadata = name##_metadata();\
  return create_generated_subgraph_impl(ranks, dims, metadata, name##_argv);\
}\
\
void free_##name(ConfiguredCompiledFunc *func)\
{\
  delete [] func->configuration;\
}

extern "C"
{

GENERATED_OPERATOR(generated_subgraph_0)

} // extern "C"

// ************************************ test part

#include <iostream>
#include <chrono>

void fill_sequential(float *data, int n)
{
  for (int i = 0; i < n; ++i)
    data[i] = i;
}

int main()
{
  int ranks[] = {2,2,2,2};

  constexpr int N = 384;
  constexpr int runs = 1000;

  int shape_x[] = {1, N};
  int shape_y[] = {1, N*3};
  int shape_z[] = {1, N*3};
  int shape_output[] = {1, N};
  int *shapes[] = {shape_x, shape_y, shape_z, shape_output};
  auto func = create_generated_subgraph_0(ranks, shapes);

  float data_x[N];
  float data_y[N*3];
  float data_z[N*3];
  float data_output[N];
  fill_sequential(data_x, N);
  fill_sequential(data_y, N*3);
  fill_sequential(data_z, N*3);
  void *args[] = {data_x, data_y, data_z, data_output};

  auto start = std::chrono::system_clock::now();

  for (int i = 0; i < runs; ++i)
  {
    func.wrapper(func.configuration, args);
  }

  auto finish = std::chrono::system_clock::now();

  float elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
  std::cout << "elapsed " << elapsed / runs << " microseconds\n";

  for (int i = 0; i < 10; ++i)
  {
    std::cout << data_output[i] << "\n";
  }
  free_generated_subgraph_0(&func);
}

