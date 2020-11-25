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

#ifndef __ONERT_EXEC_JSON_EXEC_TIME_H__
#define __ONERT_EXEC_JSON_EXEC_TIME_H__

#include <fstream>
#include <unordered_map>
#include <map>
#include <vector>
#include "backend/Backend.h"
#include "backend/IConfig.h"

namespace onert
{
namespace exec
{

/**
 * @brief table, that contains execution time of an operation on some backend for different input
 * sizes and transfer time from one backend to another for various input sizes (permutation time)
 *
 *               backend ->  op ->  quant->  size   --> time
 * _measurements[Backend*]["string"][bool][uint32_t] = int64_t
 */
using MeasurementData = std::unordered_map<
    const backend::Backend *,
    std::unordered_map<std::string, std::unordered_map<bool, std::map<uint32_t, int64_t>>>>;

class JSON
{
public:
  explicit JSON(const std::vector<const backend::Backend *> &backends,
                MeasurementData &measurements)
      : _measurement_file("exec_time.json"), _backends(), _measurements(measurements)
  {
    for (const auto b : backends)
    {
      _backends.emplace(b->config()->id(), b);
    }
    loadOperationsExecTime();
  };
  /**
   * @brief Update _measurement_file with new data.
   */
  void storeOperationsExecTime() const;

private:
  ///@brief file containing measurements
  std::string _measurement_file;
  std::unordered_map<std::string, const backend::Backend *> _backends;
  MeasurementData &_measurements;

  /**
   * @brief Helper function for inserting data to OperationExecTimes
   *
   * @param backend String name of backend
   * @param operation String name of operation
   * @param quant if input type quantized
   * @param stream File stream
   */
  void readOperation(const std::string &backend, const std::string &operation, bool quant,
                     std::ifstream &stream);

  /**
   * @brief Helper function for writing OperationExecTimes to stream
   *
   * @param operation_info Map of operations execution information
   * @param stream File stream
   */
  void printOperation(const std::map<uint32_t, int64_t> &operation_info,
                      std::ofstream &stream) const;
  /**
   * @brief Parse and load _measurements from _measurement_file.
   */
  void loadOperationsExecTime();
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_JSON_EXEC_TIME_H__
