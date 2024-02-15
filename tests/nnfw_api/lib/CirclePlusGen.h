/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_API_TEST_CIRCLE_PLUS_GEN_H__
#define __NNFW_API_TEST_CIRCLE_PLUS_GEN_H__

#include "CircleGen.h"
#include "circle_traininfo_generated.h"

struct CirclePlusBuffer
{
  CircleBuffer circle;
  CircleBuffer circle_plus;
  std::vector<int> expected;
};

class CirclePlusGen : public CircleGen
{
public:
  CirclePlusGen() = default;

  struct TrainInfo
  {
    circle::Optimizer optimizer;
    float learning_rate;
    circle::LossFn loss_fn;
    circle::LossReductionType loss_reduction_type;
    int32_t batch_size;
  };

public:
  void addTrainInfo(const TrainInfo &info);

  // NOTE: this is overriden from CircleGen::finish()
  CirclePlusBuffer finish();

private:
  CircleBuffer createModelTraining();

private:
  flatbuffers::FlatBufferBuilder _fbb_plus{1024};
  TrainInfo _info;
};

#endif // __NNFW_API_TEST_CIRCLE_PLUS_GEN_H__
