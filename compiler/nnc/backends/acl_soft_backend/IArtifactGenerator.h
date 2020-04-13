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

#ifndef _NNC_ARTIFACT_GENERATOR_INTERFACE_H_
#define _NNC_ARTIFACT_GENERATOR_INTERFACE_H_

namespace nnc
{

class ArtifactEntity;
class ArtifactLiteral;
class ArtifactNamed;
class ArtifactVariable;
class ArtifactExpr;
class ArtifactId;
class ArtifactRef;
class ArtifactDeref;
class ArtifactFunctionCall;
class ArtifactUnaryExpr;
class ArtifactBinaryExpr;
class ArtifactIndex;
class ArtifactRet;
class ArtifactBreak;
class ArtifactCont;
class ArtifactBlock;
class ArtifactForLoop;
class ArtifactIf;
class ArtifactFunction;
class ArtifactModule;
class ArtifactClass;
class ArtifactClassMember;
class ArtifactClassVariable;
class ArtifactClassFunction;

/**
 * @brief The interface of the artifact source code producer.
 */
class IArtifactGenerator
{
public:
  virtual ~IArtifactGenerator() = default;

  virtual void visit(const ArtifactLiteral *node) = 0;
  virtual void visit(const ArtifactId *node) = 0;
  virtual void visit(const ArtifactRef *node) = 0;
  virtual void visit(const ArtifactDeref *node) = 0;
  virtual void visit(const ArtifactVariable *node) = 0;
  virtual void visit(const ArtifactFunctionCall *node) = 0;
  virtual void visit(const ArtifactUnaryExpr *node) = 0;
  virtual void visit(const ArtifactBinaryExpr *node) = 0;
  virtual void visit(const ArtifactIndex *node) = 0;
  virtual void visit(const ArtifactRet *node) = 0;
  virtual void visit(const ArtifactBreak *node) = 0;
  virtual void visit(const ArtifactCont *node) = 0;
  virtual void visit(const ArtifactBlock *node) = 0;
  virtual void visit(const ArtifactForLoop *node) = 0;
  virtual void visit(const ArtifactIf *node) = 0;
  virtual void visit(const ArtifactFunction *node) = 0;
  virtual void visit(const ArtifactClass *node) = 0;
  virtual void visit(const ArtifactClassVariable *node) = 0;
  virtual void visit(const ArtifactClassFunction *node) = 0;
  virtual void visit(const ArtifactModule *node) = 0;
};

} // namespace nnc

#endif //_NNC_ARTIFACT_GENERATOR_INTERFACE_H_
