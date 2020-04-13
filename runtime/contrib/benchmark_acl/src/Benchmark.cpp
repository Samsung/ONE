/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Benchmark.h"

#include <cstdlib>

Count::Count() : _value(1)
{
  auto env = std::getenv("COUNT");

  if (env)
  {
    _value = std::strtol(env, NULL, 0);
  }
}

uint32_t Count::value(void) const { return _value; }

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>

#include <iostream>
#include <chrono>

using namespace boost::accumulators;

void run_benchmark(arm_compute::graph::frontend::Stream &graph)
{
  // NOTE Here the number of warming-up iterations is hardcoded
  // TODO Decide the number of warming-up iterations appropriately
  for (uint32_t n = 0; n < 3; ++n)
  {
    auto beg = std::chrono::steady_clock::now();
    graph.run();
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);

    std::cout << "Warming-up " << n << ": " << elapsed.count() << "ms" << std::endl;
  }

  accumulator_set<double, stats<tag::mean>> acc;

  const Count count;

  for (uint32_t n = 0; n < count.value(); ++n)
  {
    auto beg = std::chrono::steady_clock::now();
    graph.run();
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);

    std::cout << "Iteration " << n << ": " << elapsed.count() << "ms" << std::endl;

    acc(elapsed.count());
  }

  std::cout << "--------" << std::endl;
  std::cout << "Mean: " << mean(acc) << "ms" << std::endl;
}
