/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_IMPORT_H__
#define __CIRCLE_IMPORT_H__

#include <mio/circle/schema_generated.h>

#include <circlechef.pb.h>

#include <map>
#include <vector>

namespace circlechef
{

using CircleSubGraphs_t = flatbuffers::Vector<flatbuffers::Offset<circle::SubGraph>>;
using CircleTensors_t = flatbuffers::Vector<flatbuffers::Offset<circle::Tensor>>;
using CircleBuffers_t = flatbuffers::Vector<flatbuffers::Offset<circle::Buffer>>;
using CircleOperators_t = flatbuffers::Vector<flatbuffers::Offset<circle::Operator>>;

const char *tensor_type(const circle::Tensor *tensor);
const char *tensor_name(const circle::Tensor *tensor);
bool is_valid(const circle::OperatorCode *opcode);
bool is_custom(const circle::OperatorCode *opcode);

/**
 * @brief Loads TF lite file and provides helpers to access attributes
 */
class CircleImport
{
public:
  CircleImport(const circle::Model *model);

  CircleImport() = delete;

public:
  bool select_sub_graph(uint32_t subgraph);

public:
  const CircleBuffers_t *buffers() { return _buffers; }
  const CircleTensors_t *tensors() { return _tensors; }
  const CircleOperators_t *operators() { return _operators; }
  const std::vector<int32_t> &inputs() const { return _inputs; }
  const std::vector<int32_t> &outputs() const { return _outputs; }

  uint32_t num_subgraph() const { return _subgraphs->Length(); }

  circle::BuiltinOperator builtin_code(const circle::Operator *op) const;
  std::string opcode_name(const circle::Operator *op) const;
  size_t buffer_info(const circle::Tensor *tensor, const uint8_t **buff_data);

  /**
   * @brief This will record the tensor by index, if it needs filler option,
   *        such as kernel, bias.
   */
  void set_tensor_filler(uint32_t tensor_index) { _tensor_filler[tensor_index] = true; }

  /**
   * @brief This will store int32 filler values such as reshape information for the tensor
   */
  void set_tensor_filler(uint32_t tensor_index, std::vector<int32_t> &expvalues)
  {
    _tensor_filler_vint32[tensor_index] = expvalues;
  }

  void set_tensor_filler(uint32_t tensor_index, std::vector<float> &expvalues)
  {
    _tensor_filler_vfloat[tensor_index] = expvalues;
  }

  /**
   * @brief This will return true if the tensor by index, needs a filler option.
   */
  bool get_tensor_filler(uint32_t tensor_index)
  {
    auto it = _tensor_filler.find(tensor_index);
    if (it != _tensor_filler.end())
    {
      return it->second;
    }
    return false;
  }

  /**
   * @brief This will return true if the tensor by index, needs a int array filler option.
   */
  bool get_tensor_filler(uint32_t tensor_index, std::vector<int32_t> &expvalues)
  {
    auto it = _tensor_filler_vint32.find(tensor_index);
    if (it != _tensor_filler_vint32.end())
    {
      expvalues = it->second;
      return true;
    }
    return false;
  }

  bool get_tensor_filler(uint32_t tensor_index, std::vector<float> &expvalues)
  {
    auto it = _tensor_filler_vfloat.find(tensor_index);
    if (it != _tensor_filler_vfloat.end())
    {
      expvalues = it->second;
      return true;
    }
    return false;
  }

private:
  const CircleSubGraphs_t *_subgraphs{nullptr};
  const CircleBuffers_t *_buffers{nullptr};
  const CircleTensors_t *_tensors{nullptr};
  const CircleOperators_t *_operators{nullptr};

  std::vector<const circle::OperatorCode *> _op_codes{};
  std::vector<int32_t> _inputs{};
  std::vector<int32_t> _outputs{};

  std::map<uint32_t, bool> _tensor_filler{};
  std::map<uint32_t, std::vector<int32_t>> _tensor_filler_vint32{};
  std::map<uint32_t, std::vector<float>> _tensor_filler_vfloat{};
};

} // namespace circlechef

#endif // __CIRCLE_IMPORT_H__
