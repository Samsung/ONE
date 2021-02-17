#include "generated_subgraph_0.h"

extern "C"
{

/**
 Full description of one input/output buffer
 */
struct alignas(void*) HalideBuffer
{
  halide_dimension_t dims[4]; // 4 is current max number of dimensions supported by Halide
  halide_buffer_t buffer_data;
};

struct alignas(void*) HalideConfiguration
{
  int num_arguments;
  void **args;
  HalideBuffer *buffers;
};

void generated_subgraph_0_impl(char *configuration, void **args)
{
  auto *h_config = reinterpret_cast<HalideConfiguration *>(configuration);
  for (int i = 0; i < h_config->num_arguments; ++i)
  {
    h_config->buffers[i].buffer_data.host = static_cast<uint8_t *>(args[i]);
  }
  generated_subgraph_0_argv(h_config->args);
}

typedef void (*CompiledFuncPtr)(char *configuration, void **); // accepts array of pointers to buffers with data

struct ConfiguredCompiledFunc
{
  CompiledFuncPtr func;
  char *configuration;
};

ConfiguredCompiledFunc create_generated_subgraph_0(int arguments, int ranks[], int *dims[])
{
  ConfiguredCompiledFunc func;
  func.func = generated_subgraph_0_impl;

  const halide_filter_metadata_t *metadata = generated_subgraph_0_metadata();

/*
  Allocate and initilize memory for Halide structures.
  First place HalideConfiguration structure, then array of HalideBuffers, then array of void* that will be passed to generated function.
*/

  int need_memory = sizeof(HalideConfiguration) + arguments * (sizeof(HalideBuffer) + arguments*sizeof(void*));
  char *raw_config = new char[need_memory];
  func.configuration = raw_config;
  
  auto *header = reinterpret_cast<HalideConfiguration *>(raw_config);
  auto *buffers = reinterpret_cast<HalideBuffer *>(raw_config + sizeof(HalideConfiguration));
  auto *args = reinterpret_cast<void **>(raw_config + sizeof(HalideConfiguration) + arguments * sizeof(HalideBuffer));

  header->num_arguments = arguments;
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

void free_generated_subgraph_0(ConfiguredCompiledFunc *func)
{
  delete [] func->configuration;
}

} // extern "C"

int main()
{
//  void *args[] = {bb.raw_buffer()};
//  generated_subgraph_0_argv(args);
  int ranks[] = {2,2,2,2};
  int shape_x[] = {1, 384};
  int shape_y[] = {1, 1152};
  int shape_z[] = {1, 1152};
  int shape_output[] = {1, 384};
  int *shapes[] = {shape_x, shape_y, shape_z, shape_output};
  auto func = create_generated_subgraph_0(4, ranks, shapes);
  // todo run
  free_generated_subgraph_0(&func);
}

