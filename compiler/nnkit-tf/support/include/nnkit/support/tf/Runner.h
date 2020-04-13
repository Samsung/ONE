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

#ifndef __NNKIT_SUPPORT_TF_RUNNER_H__
#define __NNKIT_SUPPORT_TF_RUNNER_H__

#include "nnkit/support/tftestinfo/ParsedTensor.h"
#include "nnkit/support/tf/TensorDataMap.h"
#include <angkor/TensorShape.h>

#include <tensorflow/c/c_api.h>

#include <vector>

namespace nnkit
{
namespace support
{
namespace tf
{

using nnkit::support::tftestinfo::ParsedTensor;

class Runner final
{
public:
  enum class DataType
  {
    Unknown, // Unknown type (serves as a default value)

    U8,  // 8-bit unsigned integer
    U16, // 16-bit unsigned integer
    U32, // 32-bit unsigned integer
    U64, // 64-bit unsigned integer

    S8,  // 8-bit signed integer
    S16, // 16-bit signed integer
    S32, // 32-bit signed integer
    S64, // 64-bit signed integer

    FLOAT, // floating-point
  };

public:
  Runner(const char *pb_path);

  ~Runner();

  /**
   * @brief Get tensor shape from GraphDef for input tensor only.
   *
   * @note If the node cannot be found or shape you provided is wrong or not enough though shape
   * must be needed because of unknown shape in GraphDef, it returns false.
   */
  bool getTensorShapeFromGraphDef(const std::unique_ptr<ParsedTensor> &tensor,
                                  angkor::TensorShape &shape);

  /**
   * @brief Get tensor data type from GraphDef.
   *
   * @note If the node cannot be found or dtype of the node is unknown, it returns false.
   */
  bool getTensorDtypeFromGraphDef(const std::unique_ptr<ParsedTensor> &tensor,
                                  Runner::DataType &dtype);

  void prepareInputs(const std::vector<std::unique_ptr<ParsedTensor>> &inputs,
                     TensorDataMap &data_map);

  void prepareOutputs(const std::vector<std::unique_ptr<ParsedTensor>> &outputs);

  void run();

  const std::vector<TF_Tensor *> &output() { return _output_tensors; }

private:
  TF_Graph *_graph;
  TF_Session *_sess;

  std::vector<TF_Output> _input_ops;
  std::vector<TF_Tensor *> _input_tensors;

  std::vector<TF_Output> _output_ops;
  std::vector<TF_Tensor *> _output_tensors;

  TF_Status *_status;
};

} // namespace tf
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_TF_RUNNER_H__
