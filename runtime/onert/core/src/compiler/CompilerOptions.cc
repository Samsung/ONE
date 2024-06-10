/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "compiler/CompilerOptions.h"

#include "../backend/builtin/Backend.h"

#include "util/ConfigSource.h"
#include "util/logging.h"

#include <misc/string_helpers.h>

namespace
{

using namespace onert;

std::string getOpBackends(std::unordered_map<ir::OpCode, std::string> &opcode_to_backend)
{
  std::unordered_map<ir::OpCode, std::string>::iterator it;
  std::string opbackends;

  for (it = opcode_to_backend.begin(); it != opcode_to_backend.end(); ++it)
  {
    if (!opbackends.empty())
      opbackends = opbackends + ", ";

    auto opcode = it->first;
    const std::string opname = ir::toString(opcode);
    opbackends += opname + "=" + it->second;
  }
  return opbackends;
}

} // namespace

namespace onert
{
namespace compiler
{

void ManualSchedulerOptions::setBackendMap(const std::string &str)
{
  // TODO Support multiple subgraphs for manual scheduling
  auto key_val_list = nnfw::misc::split(str, ';');
  for (const auto &key_val_str : key_val_list)
  {
    if (key_val_str.empty())
    {
      continue;
    }

    auto key_val = nnfw::misc::split(key_val_str, '=');
    const auto &key_str = key_val.at(0);
    const auto &val = key_val.at(1);
    auto key = static_cast<uint32_t>(std::stoi(key_str));
    this->index_to_backend.emplace(ir::OperationIndex{key}, val);
  }
}

std::unique_ptr<CompilerOptions> CompilerOptions::fromGlobalConfig()
{
  auto o = std::make_unique<CompilerOptions>();
  o->backend_list = nnfw::misc::split(util::getConfigString(util::config::BACKENDS), ';');
  o->minmax_dump = util::getConfigBool(util::config::MINMAX_DUMP);
  o->tracing_mode = util::getConfigBool(util::config::TRACING_MODE);
  o->graph_dump_level = util::getConfigInt(util::config::GRAPH_DOT_DUMP);
  o->executor = util::getConfigString(util::config::EXECUTOR);
  o->he_scheduler = util::getConfigBool(util::config::USE_SCHEDULER);
  o->he_profiling_mode = util::getConfigBool(util::config::PROFILING_MODE);
  o->fp16_enable = util::getConfigBool(util::config::FP16_ENABLE);
  o->workspace_dir = util::getConfigString(util::config::WORKSPACE_DIR);
  {
    // Backend for all
    auto &ms_options = o->manual_scheduler_options;

    // Default value for op_backend_all is first element in the backend list
    ms_options.backend_for_all = util::getConfigString(util::config::OP_BACKEND_ALLOPS);

// Opcode to Backend
#define OP(OpName)                                                                      \
  {                                                                                     \
    const auto &backend_str = util::getConfigString(util::config::OP_BACKEND_##OpName); \
    if (!backend_str.empty())                                                           \
    {                                                                                   \
      ms_options.opcode_to_backend[ir::OpCode::OpName] = backend_str;                   \
    }                                                                                   \
  }
#include "ir/Operations.lst"
#undef OP

    // Index to Backend
    auto map_str = util::getConfigString(util::config::OP_BACKEND_MAP);
    ms_options.setBackendMap(map_str);
  }
  return o;
}

void CompilerOptions::forceInternalOptions()
{
  // Set control flow backend for control flow operators
  auto &builtin_id = backend::builtin::Config::ID;
  manual_scheduler_options.opcode_to_backend[ir::OpCode::If] = builtin_id;
  manual_scheduler_options.opcode_to_backend[ir::OpCode::While] = builtin_id;
  manual_scheduler_options.opcode_to_backend[ir::OpCode::Permute] = builtin_id;

  // FIXME This is a workaround for bcq operations, should remove it
  manual_scheduler_options.opcode_to_backend[ir::OpCode::BCQFullyConnected] = "bcq";
  manual_scheduler_options.opcode_to_backend[ir::OpCode::BCQGather] = "bcq";

  // FIXME This is a workaround for bulk operations, should remove it
  manual_scheduler_options.opcode_to_backend[ir::OpCode::Bulk] = "trix";
}

void CompilerOptions::verboseOptions()
{
  VERBOSE(Compiler) << std::boolalpha << "==== Compiler Options ====" << std::endl;
  VERBOSE(Compiler) << "backend_list             : "
                    << nnfw::misc::join(backend_list.begin(), backend_list.end(), "/") << std::endl;
  VERBOSE(Compiler) << "tracing_mode             : " << tracing_mode << std::endl;
  VERBOSE(Compiler) << "graph_dump_level         : " << graph_dump_level << std::endl;
  VERBOSE(Compiler) << "executor                 : " << executor << std::endl;
  VERBOSE(Compiler) << "manual backend_for_all   : " << manual_scheduler_options.backend_for_all
                    << std::endl;
  VERBOSE(Compiler) << "manual_scheduler_options : "
                    << getOpBackends(manual_scheduler_options.opcode_to_backend) << std::endl;
  VERBOSE(Compiler) << "he_scheduler             : " << he_scheduler << std::endl;
  VERBOSE(Compiler) << "he_profiling_mode        : " << he_profiling_mode << std::endl;
  VERBOSE(Compiler) << "fp16_enable              : " << fp16_enable << std::endl
                    << std::noboolalpha;
}

} // namespace compiler
} // namespace onert
