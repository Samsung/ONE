/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "cker/operation/optimized/BatchMatMul.h"

#include <gtest/gtest.h>
#include <benchmark/benchmark.h>

#include <algorithm>
#include <Eigen/Core>

static void cker_bmm(benchmark::State &state)
{
  nnfw::cker::Shape input_shape1{1, 1, 3, 100, 2000};
  nnfw::cker::Shape input_shape2{1, 1, 3, 2000, 150};
  nnfw::cker::Shape output_shape{1, 1, 3, 100, 150};

  std::vector<float> input1;
  input1.reserve(3 * 100 * 2000);
  std::generate(std::begin(input1), std::end(input1), []() { return rand(); });
  std::vector<float> input2;
  std::generate(std::begin(input2), std::end(input2), []() { return rand(); });
  input2.reserve(3 * 2000 * 150);
  std::vector<float> output;
  output.reserve(3 * 100 * 150);

  nnfw::cker::BatchMatMulParams params{input_shape1, input_shape2};
  params.storage_order = static_cast<nnfw::cker::StorageOrder>(state.range(0));

  for (auto _ : state)
    nnfw::cker::optimized::BatchMatMul(params, input1.data(), input2.data(), output.data());
}

static void eigen_cm(benchmark::State &state)
{
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> lhs =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>::Random(1000, 2000);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> rhs =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>::Random(2000, 1500);

  for (auto _ : state) 
    [[maybe_unused]] volatile const auto result = lhs * rhs;
}

static void eigen_rm(benchmark::State &state)
{
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> lhs =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Random(1000, 2000);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rhs =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Random(2000, 1500);

  for (auto _ : state)
     [[maybe_unused]] const auto result = lhs * rhs;
}

static void eigen_mix(benchmark::State &state)
{
  const auto common_dim = (rand() % 5000) + 2000;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> lhs =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Random(5000, common_dim);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> rhs =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>::Random(common_dim, 6000);

  for (auto _ : state)
    [[maybe_unused]] const auto result = lhs * rhs;
}

BENCHMARK(cker_bmm)->Arg(0)->Name("cker RowMajor")->Unit(benchmark::kMillisecond);
BENCHMARK(cker_bmm)->Arg(1)->Name("cker ColMajor")->Unit(benchmark::kMillisecond);
BENCHMARK(eigen_cm)->Name("eigen ColMajor")->Unit(benchmark::kNanosecond);
BENCHMARK(eigen_rm)->Name("eigen RowMajor")->Unit(benchmark::kNanosecond);
BENCHMARK(eigen_mix)->Name("eigen RowMajor + ColMajor")->Unit(benchmark::kNanosecond);

TEST(cker_benchmark, bmm_benchmarks) { ::benchmark::RunSpecifiedBenchmarks(); }
