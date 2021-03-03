//#include "HalideRuntime.h"
#include "HalideBuffer.h"

#include "reference.h"

#include <chrono>
#include <iostream>

extern "C"
{

int generated_subgraph_0_argv(void **);
//int generated_subgraph_0(struct halide_buffer_t *, struct halide_buffer_t *, struct halide_buffer_t *, struct halide_buffer_t *);

}

int main()
{
  constexpr int N = 384;
  Halide::Runtime::Buffer<float> x(N, 1);
  Halide::Runtime::Buffer<float> y(N*3, 1);
  Halide::Runtime::Buffer<float> z(N*3, 1);
  Halide::Runtime::Buffer<float> output(N, 1);

  for (int i = 0; i < N; ++i)
    x.data()[i] = i;

  for (int i = 0; i < N*3; ++i)
  {
    y.data()[i] = i;
    z.data()[i] = i;
  }

  void *args[] = {x.raw_buffer(), y.raw_buffer(), z.raw_buffer(), output.raw_buffer()};

  constexpr int runs = 100000;

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < runs; ++i)
  {
    int res = generated_subgraph_0_argv(args);
    assert(res == 0);
  }
  auto finish = std::chrono::system_clock::now();

  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
  std::cout << "elapsed " << static_cast<float>(elapsed) / runs << " us" << "\n";

  bool failed = false;
  for (int i = 0; i < N; ++i)
    if (std::abs(ref_data[i] - output.data()[i]) > 1e-5)
    {
      std::cerr << "mismatch on " << i << " pos\n";
      failed = true;
    }

  if (failed)
  {
    for (int i = 0; i < 10; ++i)
      std::cout << output.data()[i] << "\n";
    return 1;
  }
  return 0;
}
