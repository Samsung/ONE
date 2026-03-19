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

#ifndef __OP_COP_CALL_H__
#define __OP_COP_CALL_H__

#include <moco/tf/Frontend.h>

#include <moco/Import/GraphBuilder.h>

namespace moco
{
namespace tf
{

/**
 * @brief GraphBuilder for COpCall node
 */
class COpCallGraphBuilder final : public GraphBuilder
{
public:
  COpCallGraphBuilder(const ModelSignature *signature) : _signature(signature)
  { /* empty */
  }
  bool validate(const tensorflow::NodeDef &) const override;
  void build(const tensorflow::NodeDef &, GraphBuilderContext *) const override;

private:
  const ModelSignature *_signature;
};

} // namespace tf
} // namespace moco

#endif // __OP_COP_CALL_H__
