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

#include "ArtifactGeneratorCppDecl.h"
#include "ArtifactModel.h"

#include <cassert>

using namespace std;

namespace nnc
{

ArtifactGeneratorCppDecl::ArtifactGeneratorCppDecl(ostream &out) : _out(out) {}

void ArtifactGeneratorCppDecl::visit(const ArtifactLiteral *node) { _out << node->getValue(); }

void ArtifactGeneratorCppDecl::visit(const ArtifactId *node) { _out << node->name(); }

void ArtifactGeneratorCppDecl::visit(const ArtifactRef *node)
{
  _out << "&";
  node->obj()->accept(this);
}

void ArtifactGeneratorCppDecl::visit(const ArtifactDeref *node)
{
  _out << "*";
  node->obj()->accept(this);
}

void ArtifactGeneratorCppDecl::visit(const ArtifactVariable *node)
{
  _out << node->typeName() << " " << node->name();
}

void ArtifactGeneratorCppDecl::visit(const ArtifactFunctionCall * /*node*/) {}

void ArtifactGeneratorCppDecl::visit(const ArtifactUnaryExpr * /*node*/) {}

void ArtifactGeneratorCppDecl::visit(const ArtifactBinaryExpr * /*node*/) {}

void ArtifactGeneratorCppDecl::visit(const ArtifactIndex * /*node*/) {}

void ArtifactGeneratorCppDecl::visit(const ArtifactRet * /*node*/) {}

void ArtifactGeneratorCppDecl::visit(const ArtifactBreak * /*node*/) {}

void ArtifactGeneratorCppDecl::visit(const ArtifactCont * /*node*/) {}

void ArtifactGeneratorCppDecl::visit(const ArtifactBlock * /*node*/) {}

void ArtifactGeneratorCppDecl::visit(const ArtifactForLoop * /*node*/) {}

void ArtifactGeneratorCppDecl::visit(const ArtifactIf * /*node*/) {}

void ArtifactGeneratorCppDecl::visit(const ArtifactFunction *node)
{
  _out << node->getRetTypeName() << " " << node->name() << "(";

  bool add_comma = false;

  for (const auto &par : node->getParameters())
  {
    if (add_comma)
      _out << ", ";

    par->accept(this);
    add_comma = true;
  }

  _out << ");";
}

void ArtifactGeneratorCppDecl::visit(const ArtifactClass *node)
{
  _out << "class " << node->name() << " {" << endl;
  _out << "public:" << endl;
  ++_ind;

  // Generate a public default constructor here.
  _out << _ind << node->name() << "();" << endl;

  // Then generate the other stuff.

  for (const auto &e : node->publicFunctions())
  {
    _out << _ind;
    e->accept(this);
  }

  _out << endl << "private:" << endl;

  for (const auto &e : node->privateFunctions())
  {
    _out << _ind;
    e->accept(this);
  }

  if (!node->privateFunctions().empty())
    _out << endl;

  // TODO add public variables

  for (const auto &e : node->privateVariables())
  {
    _out << _ind;
    e->accept(this);
  }

  --_ind;
  _out << "};" << endl;
}

void ArtifactGeneratorCppDecl::visit(const ArtifactClassVariable *node)
{
  _out << node->typeName() << " " << node->name() << ";" << endl;
}

void ArtifactGeneratorCppDecl::visit(const ArtifactClassFunction *node)
{
  _out << node->getRetTypeName();

  if (!node->getRetTypeName().empty())
    _out << " ";

  _out << node->name() << "(";
  bool add_comma = false;

  for (const auto &p : node->getParameters())
  {
    if (add_comma)
      _out << ", ";

    p->accept(this);
    add_comma = true;
  }

  _out << ");" << endl;
}

void ArtifactGeneratorCppDecl::visit(const ArtifactModule *node)
{
  for (const auto &i : node->headerSysIncludes())
    _out << "#include <" << i << ">" << endl;

  if (!node->headerSysIncludes().empty())
    _out << endl;

  for (const auto &i : node->headerIncludes())
    _out << "#include \"" << i << "\"" << endl;

  if (!node->headerIncludes().empty())
    _out << endl;

  for (const auto &e : node->entities())
    e->accept(this);
}

} // namespace nnc
