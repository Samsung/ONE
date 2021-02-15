#include <chrono>
#include <iostream>
#include <cmath>
#include <array>

#include "HalideBuffer.h"
#include "arithmetic.o.h"

#include "reference.h"

#define NN (100)

template <typename T> inline void init_buffer(Halide::Runtime::Buffer<T> &buf, T val)
{
  for (int z = 0; z < buf.channels(); z++)
  {
    for (int y = 0; y < buf.height(); y++)
    {
      for (int x = 0; x < buf.width(); x++)
      {
        buf(x, y, z) = val;
      }
    }
  }
}

int main()
{
  Halide::Runtime::Buffer<uint8_t> A_buf(NN, NN);
  Halide::Runtime::Buffer<uint8_t> B_buf(NN, NN);
  // Initialize matrices with pseudorandom values:
  for (int i = 0; i < NN; i++) {
      for (int j = 0; j < NN; j++) {
          A_buf(j, i) = (i + 3) * (j + 1);
          B_buf(j, i) = (i + 1) * j + 2;
      }
  }

  // Output
  Halide::Runtime::Buffer<uint8_t> C1_buf(NN, NN);

  // Reference matrix multiplication
  Halide::Runtime::Buffer<uint8_t> C2_buf(NN, NN);
  init_buffer(C2_buf, (uint8_t)0);
  for (int i = 0; i < NN; i++) {
      for (int j = 0; j < NN; j++) {
          for (int k = 0; k < NN; k++) {
              // Note that indices are flipped (see tutorial 2)
              C2_buf(j, i) += A_buf(k, i) * B_buf(j, k);
          }
      }
  }

  for (int i = 0; i < 10; ++i)
  {
    arithmetic(A_buf.raw_buffer(), B_buf.raw_buffer(), C1_buf.raw_buffer());
  }

  for (int i = 0; i < NN * NN; ++i)
    if (std::abs(C1_buf(i) - C2_buf(i)) > 1e-5)
      std::cout << "mismatch on position: " << i << "   got " << C1_buf(i) << ", but should be " << C2_buf(i) << "\n";

  constexpr int runs = 1000;
  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < runs; ++i)
  {
    arithmetic(A_buf.raw_buffer(), B_buf.raw_buffer(), C1_buf.raw_buffer());
  }
  auto finish = std::chrono::system_clock::now();
  float estimated_time = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
  std::cout << estimated_time / runs << "ms\n";
}
