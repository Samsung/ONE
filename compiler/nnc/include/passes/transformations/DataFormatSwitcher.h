/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef DATA_FORMAT_SWITCHER_PASS_H
#define DATA_FORMAT_SWITCHER_PASS_H

#include "mir/Graph.h"
#include "mir/DataFormat.h"
#include "mir/Visitor.h"

#include "pass/Pass.h"

namespace nnc
{

class DataFormatSwitcher : public Pass
{
public:
  explicit DataFormatSwitcher(mir::DataFormat target_format);

  PassData run(PassData data) override;

  void cleanup() override;

  ~DataFormatSwitcher() override;

  std::string getName() override { return "DataFormatSwitcher"; }

private:
  // operations with DataFormat dependency
  void switchAvgPool2D(mir::ops::AvgPool2DOp *op);
  void switchConv2D(mir::ops::Conv2DOp *op);
  void switchDeConv2D(mir::ops::DeConv2DOp *op);
  void switchDepthwiseConv2D(mir::ops::DepthwiseConv2DOp *op);
  void switchMaxPool2D(mir::ops::MaxPool2DOp *op);

  // helper functions
  mir::Operation::Output *insertTransposeBefore(mir::Operation::Output *out);
  mir::Operation::Output *insertTransposeAfter(mir::Operation::Output *out);

private:
  mir::Graph *_graph;
  mir::DataFormat _target_format;
  std::vector<mir::Operation *> _candidates_for_switch;
};

} // namespace nnc

#endif // DATA_FORMAT_SWITCHER_PASS_H
