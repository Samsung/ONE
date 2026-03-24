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

#ifndef _NNC_ARTIFACT_GENERATOR_CPP_DECL_H_
#define _NNC_ARTIFACT_GENERATOR_CPP_DECL_H_

#include "IArtifactGenerator.h"
#include "ArtifactIndent.h"

#include <ostream>

namespace nnc
{

/**
 * @brief The ACL C++ artifact header file producer.
 */
class ArtifactGeneratorCppDecl : public IArtifactGenerator
{
public:
  explicit ArtifactGeneratorCppDecl(std::ostream &out);

  void visit(const ArtifactLiteral *node) override;
  void visit(const ArtifactId *node) override;
  void visit(const ArtifactRef *node) override;
  void visit(const ArtifactDeref *node) override;
  void visit(const ArtifactVariable *node) override;
  void visit(const ArtifactFunctionCall *node) override;
  void visit(const ArtifactUnaryExpr *node) override;
  void visit(const ArtifactBinaryExpr *node) override;
  void visit(const ArtifactIndex *node) override;
  void visit(const ArtifactRet *node) override;
  void visit(const ArtifactBreak *node) override;
  void visit(const ArtifactCont *node) override;
  void visit(const ArtifactBlock *node) override;
  void visit(const ArtifactForLoop *node) override;
  void visit(const ArtifactIf *node) override;
  void visit(const ArtifactFunction *node) override;
  void visit(const ArtifactClass *node) override;
  void visit(const ArtifactClassVariable *node) override;
  void visit(const ArtifactClassFunction *node) override;
  void visit(const ArtifactModule *node) override;

private:
  std::ostream &_out;
  ArtifactIndent _ind;
};

} // namespace nnc

#endif //_NNC_ARTIFACT_GENERATOR_CPP_DECL_H_
