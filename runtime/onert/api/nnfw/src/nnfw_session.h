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

#ifndef __API_NNFW_SESSION_H__
#define __API_NNFW_SESSION_H__

#include "nnfw.h"

#include "CustomKernelRegistry.h"
#include "compiler/CompilerOptions.h"
#include "compiler/ICompiler.h"
#include "exec/Execution.h"
#include "ir/NNPkg.h"
#include "ir/train/TrainingInfo.h"
#include "odc/CodegenManager.h"
#include "odc/QuantizeManager.h"

#include <util/TracingCtx.h>

#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>

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

  enum class AutoCompilationState
  {
    INITIAL_STATE,          //< Initial state
    QUANTIZED_MODEL_LOADED, //< Qunatized model is loaded
    COMPILED_MODEL_LOADED   //< Compiled model is loaded
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
  NNFW_STATUS load_model_from_path(const char *path);
  NNFW_STATUS prepare();
  NNFW_STATUS run();

  NNFW_STATUS run_async();
  NNFW_STATUS await();

  NNFW_STATUS set_input(uint32_t index, NNFW_TYPE type, const void *buffer, size_t length);
  NNFW_STATUS set_output(uint32_t index, NNFW_TYPE type, void *buffer, size_t length);

  NNFW_STATUS input_size(uint32_t *number);
  NNFW_STATUS output_size(uint32_t *number);

  NNFW_STATUS set_input_layout(uint32_t index, NNFW_LAYOUT layout);
  NNFW_STATUS set_output_layout(uint32_t index, NNFW_LAYOUT layout);
  NNFW_STATUS set_input_type(uint32_t index, NNFW_TYPE type);
  NNFW_STATUS set_output_type(uint32_t index, NNFW_TYPE type);

  NNFW_STATUS set_input_tensorinfo(uint32_t index, const nnfw_tensorinfo *ti);

  NNFW_STATUS input_tensorinfo(uint32_t index, nnfw_tensorinfo *ti);
  NNFW_STATUS output_tensorinfo(uint32_t index, nnfw_tensorinfo *ti);

  NNFW_STATUS set_available_backends(const char *backends);

  NNFW_STATUS set_workspace(const char *dir);

  NNFW_STATUS set_signature_for_tensorinfo(const char *signature);
  NNFW_STATUS set_signature_run(const char *signature);

  static NNFW_STATUS deprecated(const char *msg);

  //
  // Internal-only API
  //

  NNFW_STATUS set_config(const char *key, const char *value);
  NNFW_STATUS get_config(const char *key, char *value, size_t value_size);
  NNFW_STATUS load_circle_from_buffer(uint8_t *buffer, size_t size);
  NNFW_STATUS get_output(uint32_t index, nnfw_tensorinfo *out_info, const void **out_buffer);

  //
  // Experimental API
  //
  NNFW_STATUS register_custom_operation(const std::string &id, nnfw_custom_eval eval_func);
  NNFW_STATUS input_tensorindex(const char *tensorname, uint32_t *index);
  NNFW_STATUS output_tensorindex(const char *tensorname, uint32_t *index);

  // Run inference with auto compilation
  NNFW_STATUS run_with_auto_compilation(const char *target, NNFW_CODEGEN_PREF pref);
  // Set odc parameter: minmax_records_count for quantization in auto compilation mode
  NNFW_STATUS set_odc_param_minmax_records_count(int minmax_records_count);
  // delete MinMax File of on-device compiler
  NNFW_STATUS delete_odc_minmax_file();

  /**
   * @brief   Set backends with string-encoded mapping from operation index to backend type
   *          (cpu, acl_cl)
   */
  NNFW_STATUS set_backends_per_operation(const char *backend_settings);

  NNFW_STATUS train_get_traininfo(nnfw_train_info *info);
  NNFW_STATUS train_set_traininfo(const nnfw_train_info *info);
  NNFW_STATUS train_prepare();
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
  NNFW_STATUS train_export_circleplus(const char *path);
  NNFW_STATUS train_import_checkpoint(const char *path);
  NNFW_STATUS train_export_checkpoint(const char *path);

  NNFW_STATUS set_quantization_type(NNFW_QUANTIZE_TYPE qtype);
  NNFW_STATUS set_quantized_model_path(const char *path);
  NNFW_STATUS quantize();

  NNFW_STATUS set_codegen_model_path(const char *path);
  NNFW_STATUS codegen(const char *target, NNFW_CODEGEN_PREF pref);

  NNFW_STATUS set_prepare_config(const NNFW_PREPARE_CONFIG key, const char *value);
  NNFW_STATUS reset_prepare_config();
  NNFW_STATUS set_execute_config(const NNFW_RUN_CONFIG key, const char *value);
  NNFW_STATUS reset_execute_config();

private:
  const onert::ir::IGraph *primary_subgraph();
  uint32_t getInputSize();
  uint32_t getOutputSize();
  NNFW_STATUS loadModelFile(const std::string &model_file_path, const std::string &model_type);

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
  std::unique_ptr<onert::ir::NNPkg> _nnpkg;
  std::unique_ptr<onert::compiler::CompilerOptions> _coptions;
  std::unique_ptr<onert::compiler::CompilerArtifact> _compiler_artifact;
  std::unique_ptr<onert::exec::Execution> _execution;
  std::shared_ptr<onert::api::CustomKernelRegistry> _kernel_registry;
  std::vector<std::thread> _threads;
  std::unique_ptr<onert::ir::train::TrainingInfo> _train_info;
  std::unique_ptr<onert::odc::QuantizeManager> _quant_manager;
  std::unique_ptr<onert::odc::CodegenManager> _codegen_manager;
  AutoCompilationState _autoCompilationState = AutoCompilationState::INITIAL_STATE;
  // Remember path to loaded original model
  // It may be used for on-device compiler / on-device training.
  //
  // If necessary, we may replace _model_path to _model_origin like:
  //
  //   union _model_origin {
  //     const char *path;
  //     const uint8 *buf;
  //   }
  std::filesystem::path _model_path;
  std::unordered_map<onert::ir::SubgraphIndex, std::string> _signature_map;
  onert::ir::SubgraphIndex _selected_signature;
};

#endif // __API_NNFW_SESSION_H__
