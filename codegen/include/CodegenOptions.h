/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef NNCC_CODEGENOPTIONS_H
#define NNCC_CODEGENOPTIONS_H

namespace luci_codegen
{

enum class ArchType
{
  Native,
  X86_32,
  X86_64,
  ARM_32,
  ARM_64
};

struct Architecture
{
  ArchType type = ArchType::Native;
  int l1_size = 16*1024;  // 16 kbytes is a conservative guess
};

enum class OS
{
  Native,
  Linux,
  Windows,
  Android
};

enum class SchedulerAlgorithm
{
  None,
  Mullapudi, // Mullapudi2016
  Li,        // Li2018
  Adams      // Adams2019
};

/**
 * @brief Options specify how to process graph and how to generate code
 */
struct CodegenOptions
{
public: // graph analysis options section

  /**
   * max size of constant buffer to inline in code in bytes
   */
  int max_inline_buffer_threshold = 1024;

public: // backend options

  /**
   * If true generate argument correctness checks
   */
  bool debug = true;

  Architecture arch;

  /**
   * Target operation system
   */
  OS os = OS::Native;

  /**
   * What scheduling algorithm to use (None for no scheduling)
   */
  SchedulerAlgorithm scheduler = SchedulerAlgorithm::None;
};

} // namespace luci_codegen

#endif // NNCC_CODEGENOPTIONS_H
