/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file Softmax benchmark
 */

#define NONIUS_RUNNER
#include <nonius/nonius_single.h++>

#include <cker/operation/SoftMax.h>

#include <vector>

//
// Parameters
//
NONIUS_PARAM(LEN, 1000);

//
// Implementations
//
NONIUS_BENCHMARK("cker::Softmax(float)", [](nonius::chronometer meter) {
  auto len = meter.param<LEN>();

  nnfw::cker::SoftmaxParams params;
  nnfw::cker::Shape shape{1, len};

  params.beta = 1.0;

  std::vector<float> input;
  std::vector<float> output;

  input.resize(len);
  output.resize(len);

  meter.measure([&](int) {
    // Run!
    nnfw::cker::Softmax(params, shape, input.data(), shape, output.data());
  });
})
