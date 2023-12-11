/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __MPQSOLVER_DUMPER_H__
#define __MPQSOLVER_DUMPER_H__

#include <luci/IR/Module.h>
#include <luci/CircleQuantizer.h>

#include <string>

namespace mpqsolver
{
namespace core
{

using LayerParam = luci::CircleQuantizer::Options::LayerParam;
using LayerParams = std::vector<std::shared_ptr<LayerParam>>;

class Dumper final
{
public:
  Dumper() = default;
  Dumper(const std::string &dir_path);

  /**
   * @brief sets model path for further usage
   */
  void setModelPath(const std::string &model_path);

  /**
   * @brief dumps mpq configuration
   * @param layers specific quantization parameters
   * @param def_dtype default quantization data type
   * @param def_granularity default granularity
   * @param step id of mpq configuration
   */
  void dumpMPQConfiguration(const LayerParams &layers, const std::string &def_dtype,
                            const std::string &def_granularity, int step) const;

  /**
   * @brief dumps final mpq configuration
   * @param layers specific quantization parameters
   * @param def_dtype default quantization data type
   * @param def_granularity default granularity
   */
  void dumpFinalMPQ(const LayerParams &layers, const std::string &def_dtype,
                    const std::string &def_granularity) const;

  /**
   * @brief dumps quantized module
   * @param layers specific quantization parameters
   * @param step id of quantized module
   */
  void dumpQuantized(luci::Module *module, uint32_t step) const;

  /**
   * @brief create file for error dumping
   */
  void prepareForErrorDumping() const;

  /**
   * @brief append error of Q8 quantization
   */
  void dumpQ8Error(float error) const;

  /**
   * @brief append error of Q16 quantization
   */
  void dumpQ16Error(float error) const;

  /**
   * @brief append error of mpq quantization
   * @param error error of quantization
   * @param step id of error
   */
  void dumpMPQError(float error, uint32_t step) const;

  /**
   * @brief dump final error
   * @param error final error of quantization
   */
  void dumpMPQError(float error) const;

private:
  void writeDataToFile(const std::string &path, const std::string &data) const;
  void dumpMPQConfiguration(const LayerParams &layers, const std::string &def_dtype,
                            const std::string &def_granularity, const std::string &path) const;
  void prepareDirectory(const std::string &dir_path) const;
  void saveCircle(luci::Module *module, std::string &path) const;
  void dumpError(float error, const std::string &tag, const std::string &path) const;
  std::string getErrorPath() const { return _dir_path + "/errors" + ".mpq.txt"; }

private:
  std::string _dir_path;
  std::string _model_path;

}; // Dumper

} // namespace core
} // namespace mpqsolver

#endif //__MPQSOLVER_DUMPER_H__
