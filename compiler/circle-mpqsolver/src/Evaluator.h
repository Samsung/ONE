/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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
#ifndef __MPQSOLVER_EVALUATOR_H__
#define __MPQSOLVER_EVALUATOR_H__

#include <luci/IR/Module.h>
#include <luci/CircleQuantizer.h>

#include <string>
#include <vector>

namespace mpqsolver
{
using ElementaryOutput = std::vector<char>;
using NNOutput = std::vector<ElementaryOutput>;
using DatasetOutput = std::vector<NNOutput>;
using LayerParam = luci::CircleQuantizer::Options::LayerParam;
using LayerParams = std::vector<std::shared_ptr<LayerParam>>;

class DatasetEvaluator
{
public:
  DatasetEvaluator(const luci::Module *ref_module, const std::string &h5file);
  ~DatasetEvaluator() = default;
  float evaluate(const std::string &def_quant, LayerParams &layer_params);
  std::unique_ptr<luci::Module> quantize(const std::string &def_quant, LayerParams &layer_params);

private:
  float evaluate(const luci::Module *module);

private:
  const luci::Module *_ref_module = nullptr;
  std::string _h5file;
  DatasetOutput _ref_output;
};

} // namespace mpqsolver

#endif //__MPQSOLVER_EVALUATOR_H__
