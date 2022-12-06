/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef ONERT_MICRO_INTERPRETER_CONFIGURE_H
#define ONERT_MICRO_INTERPRETER_CONFIGURE_H

namespace luci_interpreter
{

class InterpreterConfigure
{
public:
  void setAllocateInputValue(bool allocate_input) { _allocate_input = allocate_input; }
  bool getAllocateInputValue() const { return _allocate_input; }

  void useStaticMemoryManager(uint32_t input_buf_size, uint32_t temp_buf_size,
                              uint32_t output_buf_size)
  {
    _use_static_manager = true;
    _input_buf_size = input_buf_size;
    _temp_buf_size = temp_buf_size;
    _output_buf_size = output_buf_size;
  }

  bool isStaticManager() const { return _use_static_manager; }

private:
  bool _use_static_manager = false;
  bool _allocate_input = true;

public:
  // TODO: remove it and read these values from circle file
  uint32_t _input_buf_size = 0;
  uint32_t _temp_buf_size = 0;
  uint32_t _output_buf_size = 0;
};

} // namespace luci_interpreter

#endif // ONERT_MICRO_INTERPRETER_CONFIGURE_H
