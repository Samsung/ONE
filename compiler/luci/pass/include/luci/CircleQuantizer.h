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

#ifndef __LUCI_CIRCLE_QUANTIZER_H__
#define __LUCI_CIRCLE_QUANTIZER_H__

#include <loco.h>

#include <iterator>
#include <string>
#include <vector>

namespace luci
{

class CircleQuantizer final
{
public:
  struct Options
  {
    struct LayerParam
    {
      std::string name;
      std::string dtype;
      std::string granularity;
    };

    using LayerParams = std::vector<std::shared_ptr<LayerParam>>;

    // NOTE ...Set is not related with std::set but used as to denote
    //      multple 'set' of LayerParams.
    class LayerParamsSet
    {
    public:
      // some helper methods
      size_t size(void) const { return items.size(); }
      template <class... Args> void emplace_back(Args &&... args) { items.emplace_back(args...); }
      std::vector<LayerParams>::iterator begin() { return items.begin(); };
      std::vector<LayerParams>::iterator end() { return items.end(); };

    private:
      // store multiple set of LayerParams
      std::vector<LayerParams> items;
    };

    enum Algorithm
    {
      QuantizeDequantizeWeights,
      QuantizeWithMinMax,
      Requantize,
      CopyQuantParam,
      ForceQuantParam,
      ConvertToFakeQuantizedModel,
      QuantizeWeights,
    };

    enum AlgorithmParameters
    {
      // quantize
      Quantize_input_model_dtype,
      Quantize_output_model_dtype,
      Quantize_granularity, // layer-wise or channel-wise
      Quantize_tensor_names,
      Quantize_scales,
      Quantize_zero_points,
      Quantize_layer_params,

      // copy_quantparam
      Quantize_src_tensor_names,
      Quantize_dst_tensor_names,

      Quantize_input_type,
      Quantize_output_type,
      Quantize_TF_style_maxpool,
      Quantize_save_min_max,
    };

    virtual ~Options() = default;

    virtual void enable(Algorithm) = 0;
    virtual bool query(Algorithm) = 0;
    virtual void param(AlgorithmParameters, const std::string &) = 0;
    virtual const std::string param(AlgorithmParameters) const = 0;
    virtual void params(AlgorithmParameters, std::vector<std::string> &) = 0;
    virtual std::vector<std::string> params(AlgorithmParameters) const = 0;

    // Quantization parameters for multiple layers
    virtual void layer_params(AlgorithmParameters, LayerParams &) = 0;
    virtual LayerParams layer_params(AlgorithmParameters) const = 0;
    virtual void layer_params_set(LayerParamsSet &) = 0;
    virtual LayerParamsSet layer_params_set(void) const = 0;
  };

public:
  // TODO maybe caller can provide Options as ctor parameters
  Options *options(void);

public:
  void quantize(loco::Graph *) const;

private:
  std::unique_ptr<Options> _options;
};

} // namespace luci

#endif // __LUCI_CIRCLE_QUANTIZER_H__
