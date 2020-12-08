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

#ifndef __ONERT_EXEC_EXEC_TIME_H__
#define __ONERT_EXEC_EXEC_TIME_H__

#include "backend/Backend.h"
#include "backend/IConfig.h"
#include "JSONExecTime.h"
#include <memory>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

namespace onert
{
namespace exec
{
class ExecTime
{
public:
  explicit ExecTime(const std::vector<const backend::Backend *> &backends)
    : _json(backends, _measurements)
  {
  }

public:
  /**
   * @brief Get exec time of an operation with input size
   *        or linearly interpolated value based on size if there is no record for given size
   *
   * @param[in] backend id of a backend
   * @param[in] operation name of an operation
   * @param[in] quant if input type quantized
   * @param[in] op_size sum of operation's flattened sizes of inputs and outputs
   * @return execution time for given input sizes
   *         -1 if there are no records for given parameters (backend, op, quantization).
   */
  int64_t getOperationExecTime(const backend::Backend *backend, const std::string &operation,
                               bool quant, uint32_t op_size) const;
  /**
   * @brief Update exec time of the operation on a backend with given input size or
   *        add new entity if there is no one.
   *
   * @param[in] backend id of a backend
   * @param[in] operation name of an operation
   * @param[in] quant if input type quantized
   * @param[in] op_size sum of operation's flattened sizes of inputs and outputs
   * @param[in] time real measured value
   */
  void updateOperationExecTime(const backend::Backend *backend, const std::string &operation,
                               bool quant, uint32_t op_size, int64_t time);
  /**
   * @brief Get the permute time from one backend to another
   *
   * @param[in] from_backend
   * @param[in] to_backend
   * @param[in] quant if input type quantized
   * @param[in] op_size sum of operation's flattened sizes of inputs and outputs
   * @return permutation time for operation size
   */
  int64_t getPermuteTime(const backend::Backend *from_backend, const backend::Backend *to_backend,
                         bool quant, uint32_t op_size) const;
  /**
   * @brief Update permute time from one backend to another
   *
   * @param[in] from_backend
   * @param[in] to_backend
   * @param[in] quant if input type quantized
   * @param[in] time measured permutation time
   * @param[in] op_size sum of operation's flattened sizes of inputs and outputs
   */
  void updatePermuteTime(const backend::Backend *from_backend, const backend::Backend *to_backend,
                         bool quant, uint32_t op_size, int64_t time);
  /**
   * @brief Get the max value of int32_t in int64_t
   * @return max value
   */
  static int64_t getMax() { return _MAX; }
  /**
   * @brief Update metrics file with new data.
   */
  void storeOperationsExecTime() const { _json.storeOperationsExecTime(); }
  static const int64_t NOT_FOUND = -1;

private:
  /// @brief Measurement data, which is shared with serializer
  MeasurementData _measurements;
  // int64_t::max may cause integer overflow
  static const int64_t _MAX = std::numeric_limits<int32_t>::max();
  /// @brief Serializer
  JSON _json;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_EXEC_TIME_H__
