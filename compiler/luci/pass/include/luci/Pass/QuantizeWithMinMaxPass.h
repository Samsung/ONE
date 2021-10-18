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

#ifndef __LUCI_QUANTIZE_WITH_MINMAX_PASS_H__
#define __LUCI_QUANTIZE_WITH_MINMAX_PASS_H__

#include <loco.h>

#include <logo/Pass.h>

#include <luci/Pass/QuantizationParameters.h>

namespace luci
{

/**
 * @brief Pass to quantize activation, weights, and bias
 */
class QuantizeWithMinMaxPass : public logo::Pass
{
  // For backward-compatibility
  // TODO Remove this constructor
public:
  QuantizeWithMinMaxPass(loco::DataType input_model_dtype, loco::DataType output_model_dtype,
                         QuantizationGranularity granularity)
    : _input_model_dtype{input_model_dtype}, _output_model_dtype{output_model_dtype},
      _granularity{granularity}, _input_type{output_model_dtype}, _output_type{output_model_dtype}
  {
    // DO NOTHING
  }

public:
  QuantizeWithMinMaxPass(loco::DataType input_model_dtype, loco::DataType output_model_dtype,
                         QuantizationGranularity granularity, loco::DataType input_type,
                         loco::DataType output_type)
    : _input_model_dtype{input_model_dtype}, _output_model_dtype{output_model_dtype},
      _granularity{granularity}, _input_type{input_type}, _output_type{output_type}
  {
    // DO NOTHING
  }
  virtual const char *name(void) const { return "luci::QuantizeWithMinMaxPass"; }

public:
  bool run(loco::Graph *graph);

private:
  void set_input_type(loco::Graph *graph) const;
  void set_output_type(loco::Graph *graph) const;

private:
  loco::DataType _input_model_dtype;
  loco::DataType _output_model_dtype;
  QuantizationGranularity _granularity;
  loco::DataType _input_type;
  loco::DataType _output_type;
};

} // namespace luci

#endif //__LUCI_QUANTIZE_WITH_MINMAX_PASS_H__
