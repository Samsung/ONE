// #include <mbed.h>

#include "benchmarks/include/LuciMicroBenchmark.hpp"
extern "C" void __sync_synchronize() {}

int main()
{
  run_luci_micro_benchmark();
  return 0;
}