/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CMD_OPTIONS_H__
#define __CMD_OPTIONS_H__

namespace opts
{

inline const char *__opt_save_ops = "Save operators list instead of .circle ";
inline const char *__opt_dynamic_batch_to_single_batch =
  "Convert dynamic batch size (first dimension) of inputs to 1";
inline const char *__opt_unroll_rnn_d = "Unroll RNN Op if exist";
inline const char *__opt_unroll_lstm_d = "Unroll LSTM Op if exist";
inline const char *__opt_edbuf_d = "Tensorflow experimental_disable_batchmatmul_unfold";
inline const char *__opt_keep_io_order_d = "Rename I/O to match order (obsolete)";
inline const char *__opt_save_int_d = "Save intermediate files (obsolete)";
inline const char *__opt_check_shapeinf = "Validate shape inference";
inline const char *__opt_check_dynshapeinf = "Validate dynamic shape inference";

} // namespace opts

#endif // __CMD_OPTIONS_H__
