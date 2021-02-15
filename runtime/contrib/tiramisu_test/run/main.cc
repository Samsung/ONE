#include <chrono>
#include <iostream>
#include <cmath>
#include <array>

#include "HalideBuffer.h"
#include "arithmetic.o.h"

const int L = 4;
const int M = 100;
const int N = 8;

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
  Halide::Runtime::Buffer<uint8_t> A_buf(M, L);
  Halide::Runtime::Buffer<uint8_t> B_buf(N, M);
  // Initialize matrices with pseudorandom values:
  for (int i = 0; i < L; i++)
  {
    for (int j = 0; j < M; j++)
      A_buf(j, i) = (i + 3) * (j + 1);
  }
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
      B_buf(j, i) = (i + 1) * j + 2;
  }

  // Output
  Halide::Runtime::Buffer<uint8_t> C1_buf(N, L);

  // Reference matrix multiplication
  Halide::Runtime::Buffer<uint8_t> C2_buf(N, L);
  init_buffer(C2_buf, (uint8_t)0);
  for (int i = 0; i < L; i++)
  {
    for (int j = 0; j < N; j++)
    {
      for (int k = 0; k < M; k++)
      {
        // Note that indices are flipped (see tutorial 2)
        C2_buf(j, i) += A_buf(k, i) * B_buf(j, k);
      }
    }
  }

  for (int i = 0; i < 10; ++i)
  {
    arithmetic(A_buf.raw_buffer(), B_buf.raw_buffer(), C1_buf.raw_buffer());
  }

  for (int i = 0; i < L * N; ++i)
    if (std::abs(C1_buf(i) - C2_buf(i)) > 1e-5)
      std::cout << "mismatch on position: " << i << "   got " << (int)C1_buf(i) << ", but should be " << (int)C2_buf(i) << "\n";
    else
      std::cout << "matched  on position: " << i << "\n";

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
