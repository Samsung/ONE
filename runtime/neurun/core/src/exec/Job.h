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

#ifndef __NEURUN_EXEC_JOB_H__
#define __NEURUN_EXEC_JOB_H__

#include <unordered_set>

#include "exec/IFunction.h"
#include "ir/Index.h"
#include "ir/OperandIndexSequence.h"
#include "backend/Backend.h"

namespace neurun
{
namespace exec
{

class Job
{
public:
  /**
   * @brief Constructs a Job object
   *
   * @param index Operation index for this job
   * @param fn compiled code to run this job
   * @param inputs Input operand list
   * @param outputs Output operand list
   */
  Job(uint32_t index, IFunction *fn);
  /**
   * @brief Execute the compiled code
   */
  void run();
  /**
   * @brief Return job index
   *
   * @return Job index
   */
  uint32_t index() const { return _index; }
  /**
   * @brief Return the function to be executed
   *
   * @return Pointer of the function
   */
  IFunction *fn() { return _fn; }

private:
  uint32_t _index;
  IFunction *_fn;
};

} // namespace exec
} // namespace neurun

#endif // __NEURUN_EXEC_JOB_H__
