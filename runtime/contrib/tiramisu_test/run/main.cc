#include <chrono>
#include <iostream>
#include <cmath>
#include <array>

#include "HalideBuffer.h"
#include "arithmetic.o.h"

#include "reference.h"

int main()
{
  constexpr int runs = 100000;
  constexpr int N = 384;
  Halide::Runtime::Buffer<float> x(N);
  Halide::Runtime::Buffer<float> y(N*3);
  Halide::Runtime::Buffer<float> z(N*3);
  Halide::Runtime::Buffer<float> output(N);
  for (int i = 0; i < N; ++i)
    x(i) = i;
  for (int i = 0; i < N*3; ++i)
  {
    y(i) = i;
    z(i) = i;
  }

  for (int i = 0; i < 100; ++i)
  {
    arithmetic(x.raw_buffer(), y.raw_buffer(), z.raw_buffer(), output.raw_buffer());
  }

  for (int i = 0; i < N; ++i)
    if (std::abs(output(i) - ref_data[i]) > 1e-5)
      std::cout << "mismatch on position: " << i << "   got " << output(i) << ", but should be " << ref_data[i] << "\n";

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < runs; ++i)
  {
    arithmetic(x.raw_buffer(), y.raw_buffer(), z.raw_buffer(), output.raw_buffer());
  }
  auto finish = std::chrono::system_clock::now();
  float estimated_time = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
  std::cout << estimated_time / runs << "ms\n";
}
