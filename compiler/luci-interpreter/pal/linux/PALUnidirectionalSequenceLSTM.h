/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

#ifndef LUCI_INTERPRETER_PAL_UNIDIRECTIONALSEQUENCELSTM_H
#define LUCI_INTERPRETER_PAL_UNIDIRECTIONALSEQUENCELSTM_H

namespace luci_interpreter_pal
{

static inline void SetupScratchpadTensor(luci_interpreter::Tensor *scratchpad_3, bool use_cifg,
                                         int32_t n_batch, int n_cell)
{
  if (use_cifg)
  {
    // Reserving space for Cell, Forget, Output gates
    scratchpad_3->resize({n_batch, n_cell * 3});
  }
  else
  {
    // Reserving space for Input, Cell, Forget, Output gates
    scratchpad_3->resize({n_batch, n_cell * 4});
  }
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_UNIDIRECTIONALSEQUENCELSTM_H
