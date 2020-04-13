/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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
 * @file Rnn.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file defines RNN node
 */

#ifndef __INTERNAL_OP_RNN_H__
#define __INTERNAL_OP_RNN_H__

#include "internal/op/Node.h"

#include <cstdint>

namespace internal
{
namespace tflite
{
namespace op
{
namespace RNN
{

/**
 * @brief Struct to manipulate parameter for RNN operation
 */
struct Param
{
  int32_t output_index;           //!< index for ouuput
  int32_t hidden_state_out_index; //!< index for hidden state output

  int32_t input_index;             //!< index for input
  int32_t weights_index;           //!< index for weight
  int32_t recurrent_weights_index; //!< index for recurrent weights
  int32_t bias_index;              //!< index for bias
  int32_t hidden_state_in_index;   //!< index for hidden state input
  int32_t fused_activation_index;  //!< index for fused activation

  /**
   * @brief Default Constructor
   */
  Param() = default;
  /**
   * @brief Construct a new Param object
   * @param[in] inputCount the number of inputs
   * @param[in] inputs pointer for input data
   * @param[in] outputCount the number of outputs
   * @param[in] outputs pointer for input data
   */
  Param(uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount, const uint32_t *outputs);
};

/**
 * @brief Class to define RNN Operation
 */
class Node final : public op::Node
{
public:
  /**
   * @brief Construct a new RNN Node object
   * @param param Parameter for RNN Node
   */
  Node(const Param &param) : _param(param)
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Default Destructor
   */
  virtual ~Node() = default;

public:
  /**
   * @brief Get parameter
   * @return Param reference
   */
  const Param &param(void) const { return _param; }

public:
  /**
   * @brief Accept a NodeVisitor so that it can visit this node
   * @param [in] v Visitor
   * @return N/A
   */
  void accept(NodeVisitor &&) const override;

private:
  const Param _param; //!< parameter for RNN node
};

} // namespace RNN
} // namespace op
} // namespace tflite
} // namespace internal

#endif // __INTERNAL_OP_RNN_H__
