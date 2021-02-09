#include "Halide.h"

#include <chrono>

extern "C"
{

int generated_subgraph_0_argv(void **);
int generated_subgraph_0(struct halide_buffer_t *, struct halide_buffer_t *, struct halide_buffer_t *, struct halide_buffer_t *);

}

int main()
{
  Halide::Buffer<float> x(1, 384);
  Halide::Buffer<float> y(1, 1152);
  Halide::Buffer<float> z(1, 1152);
  Halide::Buffer<float> output(1, 384);

  for (int i = 0; i < 384; ++i)
    x.data()[i] = i;

  for (int i = 0; i < 1152; ++i)
  {
    y.data()[i] = i;
    z.data()[i] = i;
  }

  void *args[] = {x.raw_buffer(), y.raw_buffer(), z.raw_buffer(), output.raw_buffer()};

  constexpr int N = 1000000;
  int res = 0;

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < N; ++i)
  {
    res += generated_subgraph_0(x.raw_buffer(), y.raw_buffer(), z.raw_buffer(), output.raw_buffer());
  }
  auto finish = std::chrono::system_clock::now();

  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
  std::cout << "elapsed " << static_cast<float>(elapsed) / N << " us; res: " << res << "\n";

  for (int i = 0; i < 10; ++i)
    std::cout << output.data()[i] << "\n";
  return 0;
}
