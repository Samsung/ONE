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

#include "ArtifactGeneratorCppCode.h"
#include "ArtifactModel.h"

#include "AclArtifactUtilities.generated.h"

#include <cassert>

using namespace std;

namespace nnc
{

ArtifactGeneratorCppCode::ArtifactGeneratorCppCode(ostream &out) : _out(out) {}

void ArtifactGeneratorCppCode::visit(const ArtifactLiteral *node) { _out << node->getValue(); }

void ArtifactGeneratorCppCode::visit(const ArtifactId *node) { _out << node->name(); }

void ArtifactGeneratorCppCode::visit(const ArtifactRef *node)
{
  _out << "&";
  node->obj()->accept(this);
}

void ArtifactGeneratorCppCode::visit(const ArtifactDeref *node)
{
  _out << "*";
  node->obj()->accept(this);
}

void ArtifactGeneratorCppCode::visit(const ArtifactVariable *node)
{
  _out << node->typeName() << " " << node->name();

  for (const auto &d : node->getDimensions())
  {
    _out << "[";
    d->accept(this);
    _out << "]";
  }

  if (!node->getInitializers().empty())
  {
    _out << "(";
    bool add_comma = false;

    for (const auto &i : node->getInitializers())
    {
      if (add_comma)
        _out << ", ";

      i->accept(this);
      add_comma = true;
    }

    _out << ")";
  }
}

void ArtifactGeneratorCppCode::visit(const ArtifactFunctionCall *node)
{
  static const char *call_type_str[] = {".", "->", "::"};

  if (node->on())
  {
    node->on()->accept(this);
    _out << call_type_str[static_cast<int>(node->callType())];
  }

  _out << node->funcName();
  _out << "(";
  bool add_comma = false;

  for (const auto &par : node->paramList())
  {
    if (add_comma)
      _out << ", ";

    par->accept(this);
    add_comma = true;
  }

  _out << ")";
}

void ArtifactGeneratorCppCode::visit(const ArtifactUnaryExpr *node)
{
  // The trailing space is intended in new and delete!
  static const char *un_op_str[] = {"++", "--", "new ", "delete ", "++", "--"};

  if (node->getOp() < ArtifactUnOp::postIncr)
  {
    _out << un_op_str[static_cast<int>(node->getOp())];
    node->getExpr()->accept(this);
  }
  else
  {
    node->getExpr()->accept(this);
    _out << un_op_str[static_cast<int>(node->getOp())];
  }
}

void ArtifactGeneratorCppCode::visit(const ArtifactBinaryExpr *node)
{
  static const char *bin_op_str[] = {"==", "!=", "<", "<=", ">",  ">=", "=", "+",
                                     "-",  "*",  "/", "+=", "-=", "*=", "/="};
  node->getLeft()->accept(this);
  _out << " " << bin_op_str[static_cast<int>(node->getOp())] << " ";
  node->getRight()->accept(this);
}

void ArtifactGeneratorCppCode::visit(const ArtifactIndex *node)
{
  node->getExpr()->accept(this);
  _out << "[";
  node->getInd()->accept(this);
  _out << "]";
}

void ArtifactGeneratorCppCode::visit(const ArtifactRet *node)
{
  _out << "return ";
  node->expr()->accept(this);
}

void ArtifactGeneratorCppCode::visit(const ArtifactBreak * /*node*/) { _out << "break"; }

void ArtifactGeneratorCppCode::visit(const ArtifactCont * /*node*/) { _out << "continue"; }

void ArtifactGeneratorCppCode::visit(const ArtifactBlock *node)
{
  _out << " {" << endl;
  ++_ind;

  for (const auto &st : node->getStatements())
  {
    _out << _ind;
    st->accept(this);

    if (!st->isBlock())
      _out << ";" << endl;
  }

  --_ind;
  _out << _ind << "}" << endl;
}

void ArtifactGeneratorCppCode::visit(const ArtifactForLoop *node)
{
  _out << "for(";

  if (node->getInit())
    node->getInit()->accept(this);

  _out << "; ";

  if (node->getCond())
    node->getCond()->accept(this);

  _out << "; ";

  if (node->getIter())
    node->getIter()->accept(this);

  _out << ")";
  node->getBlock()->accept(this);
}

void ArtifactGeneratorCppCode::visit(const ArtifactIf *node)
{
  _out << "if(";
  node->getCond()->accept(this);
  _out << ")";
  node->getBlock()->accept(this);

  if (!node->getElseBlock()->getStatements().empty())
  {
    _out << _ind << "else";
    node->getElseBlock()->accept(this);
  }
}

void ArtifactGeneratorCppCode::visit(const ArtifactFunction * /*node*/)
{
  // TODO implement this function
}

void ArtifactGeneratorCppCode::visit(const ArtifactClass *node)
{
  // Generate a public default constructor here.
  _out << node->name() << "::" << node->name() << "()";

  if (!node->privateVariables().empty())
  {
    _out << " : ";
    bool add_delim = false;

    for (const auto &v : node->privateVariables())
    {
      if (add_delim)
        _out << ",\n";

      v->accept(this);
      add_delim = true;
    }
  }

  // TODO add constructors of public variables

  node->getConstrBlock()->accept(this);
  _out << endl;

  // Then generate the other stuff.

  for (const auto &e : node->publicFunctions())
    e->accept(this);

  for (const auto &e : node->privateFunctions())
    e->accept(this);
}

void ArtifactGeneratorCppCode::visit(const ArtifactClassVariable *node)
{
  _out << node->name() << "(";
  bool add_comma = false;

  for (const auto &i : node->getInitializers())
  {
    if (add_comma)
      _out << ", ";

    i->accept(this);
    add_comma = true;
  }

  _out << ")";
}

void ArtifactGeneratorCppCode::visit(const ArtifactClassFunction *node)
{
  _out << node->getRetTypeName();

  if (!node->getRetTypeName().empty())
    _out << " ";

  _out << node->owner()->name() << "::" << node->name() << "(";
  bool add_comma = false;

  for (const auto &p : node->getParameters())
  {
    if (add_comma)
      _out << ", ";

    p->accept(this);
    add_comma = true;
  }

  _out << ")";
  node->getBlock()->accept(this);
  _out << endl;
}

void ArtifactGeneratorCppCode::visit(const ArtifactModule *node)
{
  _out << "#include \"" << node->name() << ".h\"" << endl << endl;

  for (const auto &i : node->sourceSysIncludes())
    _out << "#include <" << i << ">" << endl;

  if (!node->sourceSysIncludes().empty())
    _out << endl;

  for (const auto &i : node->sourceIncludes())
    _out << "#include \"" << i << "\"" << endl;

  if (!node->sourceIncludes().empty())
    _out << endl;

  _out.write(AclArtifactUtilities, sizeof(AclArtifactUtilities));
  _out << endl;

  for (const auto &e : node->entities())
    e->accept(this);
}

} // namespace nnc
