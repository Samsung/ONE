/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CONV_BACKEND_H__
#define __CONV_BACKEND_H__

#include <nnsuite/conv/Model.h>
#include <nnkit/support/tflite/AbstractBackend.h>

#include <vector>

class ConvBackend final : public nnkit::support::tflite::AbstractBackend
{
public:
  explicit ConvBackend(const nnsuite::conv::Model &model);

public:
  ::tflite::Interpreter &interpreter(void) override { return _interp; }

private:
  // NOTE tflite interpreter just stores the pointer of its name
  const std::string _ifm_name;
  const std::string _ofm_name;

  // NOTE kernel data should live longer than tflite interpreter itself
  std::vector<float> _kernel;

  // NOTE bias is mssing in conv sample model, but conv op kernel in
  //      tensorflow lite interpreter does not work without bias.
  //
  //      Let's feed zero-bias as a workaround
  std::vector<float> _bias;

private:
  ::tflite::Interpreter _interp;
};

#endif // __BACKEND_H__
