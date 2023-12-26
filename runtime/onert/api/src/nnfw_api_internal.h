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

#ifndef __API_NNFW_API_INTERNAL_H__
#define __API_NNFW_API_INTERNAL_H__

#include "nnfw.h"
#include "nnfw_experimental.h"

#include <util/TracingCtx.h>

#include <string>
#include <memory>
#include <thread>
#include <vector>

namespace onert
{
namespace api
{
class CustomKernelRegistry;
} // namespace api
namespace exec
{
class Execution;
} // namespace exec
namespace ir
{
struct IGraph;
class Model;
class NNPkg;
namespace train
{
class TrainingInfo;
}
} // namespace ir
namespace compiler
{
struct CompilerArtifact;
class CompilerOptions;
} // namespace compiler
namespace odc
{
class QuantizeManager;
} // namespace odc
} // namespace onert

struct nnfw_session
{
private:
  /**
   * @brief Enum class to express the session's state
   *
   * State transition diagram:
   *
   *           +--------------+
   *           | INITIALIZED  |
   *           +--------------+
   *             |
   *             | load_model
   *             v
   *           +--------------+
   *           | MODEL_LOADED |
   *           +--------------+
   *             |
   *             | prepare
   *             v
   *           +--------------+
   *           |   PREPARED   | --------+
   *           +--------------+         |
   *             |                      |
   *             | run                  |
   *             v                      |
   *           +--------------+  run    |
   *           |              | -----+  |
   *   +-----> | FINISHED_RUN |      |  | run_async
   *   |       |              | <----+  |
   *   |       +--------------+         |
   *   |         |                      |
   *   | await   | run_async            |
   *   |         v                      |
   *   |       +--------------+         |
   *   +------ |   RUNNING    | <-------+
   *           +--------------+
   */
  enum class State
  {
    INITIALIZED,       //< Session is initialized and nothing has done to it
    MODEL_LOADED,      //< Model is loaded
    PREPARED,          //< Prepared(compiled) for execution
    RUNNING,           //< Execution is in progress (only for asynchronous execution)
    FINISHED_RUN,      //< Executed at least once
    PREPARED_TRAINING, //< Prepared for training
    FINISHED_TRAINING  //< Trained at least once
  };

public:
  /**
   * @brief Factory method. It creates and initialize nnfw_session
   *
   * @note  Use factory instead of constructor to get status
   */
  static NNFW_STATUS create(nnfw_session **session);

private:
  nnfw_session();

public:
  ~nnfw_session();
  NNFW_STATUS load_model_from_nnpackage(const char *package_file_path);
  NNFW_STATUS prepare();
  NNFW_STATUS prepare_pipeline(const char *map_file_path);
  NNFW_STATUS run();

  NNFW_STATUS run_async();
  NNFW_STATUS await();

  NNFW_STATUS set_input(uint32_t index, NNFW_TYPE type, const void *buffer, size_t length);
  NNFW_STATUS set_output(uint32_t index, NNFW_TYPE type, void *buffer, size_t length);

  NNFW_STATUS input_size(uint32_t *number);
  NNFW_STATUS output_size(uint32_t *number);

  NNFW_STATUS set_input_layout(uint32_t index, NNFW_LAYOUT layout);
  NNFW_STATUS set_output_layout(uint32_t index, NNFW_LAYOUT layout);

  NNFW_STATUS apply_tensorinfo(uint32_t index, nnfw_tensorinfo ti); // Will be deprecated
  NNFW_STATUS set_input_tensorinfo(uint32_t index, const nnfw_tensorinfo *ti);

  NNFW_STATUS input_tensorinfo(uint32_t index, nnfw_tensorinfo *ti);
  NNFW_STATUS output_tensorinfo(uint32_t index, nnfw_tensorinfo *ti);

  NNFW_STATUS set_available_backends(const char *backends);
  NNFW_STATUS set_op_backend(const char *op, const char *backend);

  //
  // Internal-only API
  //

  NNFW_STATUS set_config(const char *key, const char *value);
  NNFW_STATUS get_config(const char *key, char *value, size_t value_size);
  NNFW_STATUS load_circle_from_buffer(uint8_t *buffer, size_t size);
  NNFW_STATUS load_model_from_modelfile(const char *file_path);

  //
  // Experimental API
  //
  NNFW_STATUS push_pipeline_input(std::vector<void *> *inputs, std::vector<uint32_t> *lengths);
  NNFW_STATUS pop_pipeline_output(std::vector<void *> *outputs);

  NNFW_STATUS register_custom_operation(const std::string &id, nnfw_custom_eval eval_func);
  NNFW_STATUS input_tensorindex(const char *tensorname, uint32_t *index);
  NNFW_STATUS output_tensorindex(const char *tensorname, uint32_t *index);
  /**
   * @brief   Set backends with string-encoded mapping from operation index to backend type
   *          (cpu, acl_cl)
   */
  NNFW_STATUS set_backends_per_operation(const char *backend_settings);

  NNFW_STATUS train_prepare(const nnfw_train_info *info);
  NNFW_STATUS train_input_tensorinfo(uint32_t index, nnfw_tensorinfo *ti);
  NNFW_STATUS train_expected_tensorinfo(uint32_t index, nnfw_tensorinfo *ti);
  NNFW_STATUS train_set_input(uint32_t index, const void *input,
                              const nnfw_tensorinfo *input_tensorinfo);
  NNFW_STATUS train_set_expected(uint32_t index, const void *expected,
                                 const nnfw_tensorinfo *expected_tensorinfo);
  NNFW_STATUS train_set_output(uint32_t index, NNFW_TYPE type, void *buffer, size_t length);
  NNFW_STATUS train_run(bool update_weights);
  NNFW_STATUS train_get_loss(uint32_t index, float *loss);
  NNFW_STATUS train_export_circle(const char *path);

  NNFW_STATUS set_quantization_type(NNFW_QUANTIZE_TYPE qtype);
  NNFW_STATUS set_quantized_model_path(const char *path);
  NNFW_STATUS quantize();

private:
  const onert::ir::IGraph *primary_subgraph();
  uint32_t getInputSize();
  uint32_t getOutputSize();

  bool isStateInitialized();
  bool isStateModelLoaded();
  bool isStatePrepared();
  bool isStateRunning();
  bool isStateFinishedRun();
  bool isStatePreparedOrFinishedRun();
  bool isStatePreparedTraining();
  bool isStateFinishedTraining();
  bool isStatePreparedOrFinishedTraining();

private:
  State _state{State::INITIALIZED};
  std::shared_ptr<onert::ir::NNPkg> _nnpkg;
  std::vector<std::unique_ptr<onert::compiler::CompilerOptions>> _coptions;
  std::shared_ptr<onert::compiler::CompilerArtifact> _compiler_artifact;
  std::unique_ptr<onert::exec::Execution> _execution;
  std::shared_ptr<onert::api::CustomKernelRegistry> _kernel_registry;
  std::vector<std::thread> _threads;
  uint32_t _training_step{0};
  std::unique_ptr<onert::ir::train::TrainingInfo> _train_info;
  std::unique_ptr<onert::odc::QuantizeManager> _quant_manager;
  // Remember path to loaded original model
  // It may be used for on-device compiler / on-device training.
  //
  // If necessary, we may replace _model_path to _model_origin like:
  //
  //   union _model_origin {
  //     const char *path;
  //     const uint8 *buf;
  //   }
  std::string _model_path;
};

#endif // __API_NNFW_API_INTERNAL_H__
