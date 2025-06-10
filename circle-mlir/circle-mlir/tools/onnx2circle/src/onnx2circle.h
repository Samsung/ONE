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

#ifndef __ONNX2CIRCLE_H__
#define __ONNX2CIRCLE_H__

#include <string>

struct O2Cparam
{
  std::string sourcefile;
  std::string targetfile;

  bool save_ops = false;
  bool unroll_rnn = false;
  bool unroll_lstm = false;
  bool unfold_batchmatmul = false;
  bool check_shapeinf = false;
  bool check_dynshapeinf = false;
  // TODO add more if necessary
};

extern const char *__version;

int entry(const O2Cparam &param);

#endif // __ONNX2CIRCLE_H__
