/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_CORE_EVENTNOTIFIER_H
#define LUCI_INTERPRETER_CORE_EVENTNOTIFIER_H

namespace luci_interpreter
{

// Used at execution stage to tell the interpreter that the runtime state has changed in some way.
class EventNotifier
{
public:
  virtual ~EventNotifier() = default;

  virtual void postTensorWrite(const Tensor *tensor) = 0;
  virtual void preOperatorExecute(const Kernel *kernel) = 0;
  virtual void postOperatorExecute(const Kernel *kernel) = 0;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_CORE_EVENTNOTIFIER_H
