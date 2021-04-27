/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "ProgramCache.h"

#include <cstdint>
#include <string>

#include "ClProgram.h"
#include "Status.h"
#include "Util.h"
#include <farmhash.h>

namespace onert
{
namespace backend
{
namespace gpu_cl
{

ProgramCache::ProgramDescriptor::ProgramDescriptor(const std::string &code_text,
                                                   const std::string &options,
                                                   bool use_fingerprints)
  : code(code_text), compiler_options(options), use_fingerprint(use_fingerprints)
{
  const uint64_t code_fingerprint = ::util::Fingerprint64(code);
  const uint64_t options_fingerprint = ::util::Fingerprint64(compiler_options);
  fingerprint = code_fingerprint + options_fingerprint;
}

ProgramCache::ProgramDescriptor::ProgramDescriptor(uint64_t fingerprints)
  : fingerprint(fingerprints), use_fingerprint(true)
{
}

ProgramCache::ProgramCache(ProgramCache &&program_cache)
  : use_fingerprints_(program_cache.use_fingerprints_),
    programs_(std::move(program_cache.programs_))
{
}

ProgramCache &ProgramCache::operator=(ProgramCache &&program_cache)
{
  if (this != &program_cache)
  {
    use_fingerprints_ = program_cache.use_fingerprints_;
    programs_ = std::move(program_cache.programs_);
  }
  return *this;
}

absl::Status ProgramCache::GetOrCreateCLKernel(const std::string &code,
                                               const std::string &function_name,
                                               const std::vector<CompilerOptions> &compiler_options,
                                               const CLContext &context, const CLDevice &device,
                                               CLKernel *result)
{
  const std::string options = CompilerOptionsToString(device, compiler_options);
  ProgramDescriptor desc{code, options, use_fingerprints_};
  auto it = programs_.find(desc);
  if (it != programs_.end())
  {
    return result->CreateFromProgram(it->second, function_name);
  }

  CLProgram program;
  RETURN_IF_ERROR(CreateCLProgram(code, options, context, device, &program));
  RETURN_IF_ERROR(result->CreateFromProgram(program, function_name));
  programs_.insert(std::make_pair(std::move(desc), std::move(program)));
  return absl::OkStatus();
}

absl::Status ProgramCache::GetOrCreateCLKernel(const std::string &code,
                                               const std::string &function_name,
                                               const CLContext &context, const CLDevice &device,
                                               CLKernel *result)
{
  return GetOrCreateCLKernel(code, function_name, {}, context, device, result);
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
