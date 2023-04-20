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
  void set_model_path(const std::string &model_path);

  /**
   * @brief dumps mpq configuration
   * @param layers specific quantization parameters
   * @param def_dtype default quantization data type
   * @param param id of mpq configuration
   */
  void dump_MPQ_configuration(const LayerParams &layers, const std::string &def_dtype,
                              int param) const;

  /**
   * @brief dumps final mpq configuration
   * @param layers specific quantization parameters
   * @param def_dtype default quantization data type
   */
  void dump_final_MPQ(const LayerParams &layers, const std::string &def_dtype) const;

  /**
   * @brief dumps quantized module
   * @param layers specific quantization parameters
   * @param param id of quantized module
   */
  void dump_quantized(luci::Module *module, uint32_t param) const;

  /**
   * @brief create file for error dumping
   */
  void prepare_for_error_dumping() const;

  /**
   * @brief append error of Q8 quantization
   */
  void dump_Q8_error(float error) const;

  /**
   * @brief append error of Q16 quantization
   */
  void dump_Q16_error(float error) const;

  /**
   * @brief append error of mpq quantization
   * @param error error of quantization
   * @param id id of error
   */
  void dump_MPQ_error(float error, uint32_t param) const;

private:
  void write_data_to_file(const std::string &path, const std::string &data) const;
  void dump_MPQ_configuration(const LayerParams &layers, const std::string &def_dtype,
                              const std::string &path) const;
  void prepare_directory(const std::string &dir_path) const;
  void save_circle(luci::Module *module, std::string &path) const;
  void dump_error(float error, const std::string &tag, const std::string &path) const;
  std::string get_error_path() const { return _dir_path + "/errors" + ".mpq.txt"; }

private:
  std::string _dir_path;
  std::string _model_path;

}; // Dumper

} // namespace core
} // namespace mpqsolver

#endif //__MPQSOLVER_DUMPER_H__
