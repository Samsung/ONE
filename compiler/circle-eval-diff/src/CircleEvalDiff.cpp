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

#include "CircleEvalDiff.h"
#include "ModuleEvalDiff.h"
#include "MetricPrinter.h"

#include <foder/FileLoader.h>
#include <luci/Importer.h>

#include <stdexcept>

namespace
{

std::unique_ptr<luci::Module> import(const std::string &model_path)
{
  // Load model from the file
  foder::FileLoader loader{model_path};
  std::vector<char> model_data = loader.load();

  // Verify flatbuffers
  flatbuffers::Verifier verifier{reinterpret_cast<const uint8_t *>(model_data.data()),
                                 model_data.size()};
  if (not circle::VerifyModelBuffer(verifier))
  {
    throw std::runtime_error("Failed to verify circle '" + model_path + "'");
  }

  auto module = luci::Importer().importModule(circle::GetModel(model_data.data()));

  if (not module)
    throw std::runtime_error("Failed to load '" + model_path + "'");

  return module;
}

} // namespace

namespace circle_eval_diff
{

CircleEvalDiff::CircleEvalDiff(std::unique_ptr<Context> &&ctx)
  : _ctx(std::move(ctx)), _runner(nullptr)
{
}

CircleEvalDiff::~CircleEvalDiff() = default;

void CircleEvalDiff::init()
{
  auto first_module = import(_ctx->first_model_path);
  auto second_module = import(_ctx->second_model_path);

  // Set runner
  switch (_ctx->input_format)
  {
    case InputFormat::H5:
      _runner =
        std::make_unique<H5InputEvalDiff>(std::move(first_module), std::move(second_module));
      break;
    default:
      throw std::runtime_error("Unsupported input format.");
  }

  // Set metric
  std::unique_ptr<MetricPrinter> metric;
  for (auto metric : _ctx->metric)
  {
    switch (metric)
    {
      case Metric::MAE:
        _runner->registerMetric(std::make_unique<MAEPrinter>());
        break;
      case Metric::MAPE:
        _runner->registerMetric(std::make_unique<MAPEPrinter>());
        break;
      default:
        throw std::runtime_error("Unsupported metric.");
    }
  }
}

void CircleEvalDiff::evalDiff(const std::string &first_input_data_path,
                              const std::string &second_input_data_path) const
{
  _runner->evalDiff(first_input_data_path, second_input_data_path);
}

} // namespace circle_eval_diff
