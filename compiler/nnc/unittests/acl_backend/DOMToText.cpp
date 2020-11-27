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

/*
 * This is test set of text generator from DOM entities
 * ArtifactEntity, ArtifactNamed, ArtifactExpr and ArtifactClassMember
 * are not tested since they are abstract classes
 */

#include <sstream>
#include <tuple>

#include "ArtifactModel.h"
#include "ArtifactGeneratorCppCode.h"
#include "ArtifactGeneratorCppDecl.h"

#include "AclArtifactUtilities.generated.h"

#include "gtest/gtest.h"

using namespace std;
using namespace nnc;

using AF = ArtifactFactory;

TEST(acl_backend_dom_to_text, ArtifactLiteral)
{
  stringstream code_out;
  stringstream decl_out;
  ArtifactGeneratorCppCode code_gen(code_out);
  ArtifactGeneratorCppDecl decl_gen(decl_out);
  const char *lit_data = "hello_world";
  shared_ptr<ArtifactLiteral> lit = AF::lit(lit_data);
  lit->accept(&code_gen);
  lit->accept(&decl_gen);
  ASSERT_EQ(code_out.str(), lit_data);
  ASSERT_EQ(decl_out.str(), lit_data);
}

TEST(acl_backend_dom_to_text, ArtifactId)
{
  stringstream code_out;
  stringstream decl_out;
  ArtifactGeneratorCppCode code_gen(code_out);
  ArtifactGeneratorCppDecl decl_gen(decl_out);
  const char *id_data = "some_id";
  shared_ptr<ArtifactId> id = AF::id(id_data);
  id->accept(&code_gen);
  id->accept(&decl_gen);
  ASSERT_EQ(code_out.str(), id_data);
  ASSERT_EQ(decl_out.str(), id_data);
}

TEST(acl_backend_dom_to_text, ArtifactRef)
{
  stringstream code_out;
  stringstream decl_out;
  ArtifactGeneratorCppCode code_gen(code_out);
  ArtifactGeneratorCppDecl decl_gen(decl_out);
  const char *id_data = "some_id";
  shared_ptr<ArtifactId> id = AF::id(id_data);
  shared_ptr<ArtifactRef> ref = AF::ref(id);
  ref->accept(&code_gen);
  ref->accept(&decl_gen);
  string ref_data = string("&") + id_data;
  ASSERT_EQ(code_out.str(), ref_data);
  ASSERT_EQ(decl_out.str(), ref_data);
}

TEST(acl_backend_dom_to_text, ArtifactDeref)
{
  stringstream code_out;
  stringstream decl_out;
  ArtifactGeneratorCppCode code_gen(code_out);
  ArtifactGeneratorCppDecl decl_gen(decl_out);
  const char *id_data = "some_id";
  shared_ptr<ArtifactId> id = AF::id(id_data);
  shared_ptr<ArtifactDeref> deref = AF::deref(id);
  deref->accept(&code_gen);
  deref->accept(&decl_gen);
  string ref_data = string("*") + id_data;
  ASSERT_EQ(code_out.str(), ref_data);
  ASSERT_EQ(decl_out.str(), ref_data);
}

static void checkCall(ArtifactCallType type, const char *call_name,
                      const list<shared_ptr<ArtifactExpr>> &args, shared_ptr<ArtifactExpr> obj,
                      const char *ref_data)
{
  stringstream code_out;
  ArtifactGeneratorCppCode code_gen(code_out);
  shared_ptr<ArtifactFunctionCall> call = AF::call(call_name, args, obj, type);
  call->accept(&code_gen);
  ASSERT_EQ(code_out.str(), ref_data);
}

TEST(acl_backend_dom_to_text, ArtifactFunctionCall)
{
  const char *lit_data = "123";
  const char *id_data = "some_id";
  shared_ptr<ArtifactExpr> id = AF::id(id_data);
  shared_ptr<ArtifactExpr> lit = AF::lit(lit_data);
  const list<shared_ptr<ArtifactExpr>> args{id, lit};

  shared_ptr<ArtifactId> obj = AF::id("obj");

  using TestCase = tuple<ArtifactCallType, shared_ptr<ArtifactExpr>, const char *>;
  TestCase test_cases[] = {TestCase{ArtifactCallType::scope, nullptr, "foo(some_id, 123)"},
                           TestCase{ArtifactCallType::obj, obj, "obj.foo(some_id, 123)"},
                           TestCase{ArtifactCallType::ref, obj, "obj->foo(some_id, 123)"},
                           TestCase{ArtifactCallType::scope, obj, "obj::foo(some_id, 123)"}};

  for (const auto &test : test_cases)
  {
    ArtifactCallType call_type = get<0>(test);
    shared_ptr<ArtifactExpr> obj = get<1>(test);
    const char *ref_output = get<2>(test);
    checkCall(call_type, "foo", args, obj, ref_output);
  }
}

static void checkUnaryExpression(ArtifactUnOp op, shared_ptr<ArtifactExpr> var,
                                 const char *ref_data)
{
  stringstream code_out;
  ArtifactGeneratorCppCode code_gen(code_out);

  shared_ptr<ArtifactUnaryExpr> expr = AF::un(op, var);
  expr->accept(&code_gen);
  ASSERT_EQ(code_out.str(), ref_data);
}

TEST(acl_backend_dom_to_text, ArtifactUnaryExpr)
{
  const char *var_name = "id";
  shared_ptr<ArtifactId> var = AF::id(var_name);
  pair<ArtifactUnOp, const char *> test_cases[] = {
    {ArtifactUnOp::preIncr, "++id"},   {ArtifactUnOp::preDecr, "--id"},
    {ArtifactUnOp::heapNew, "new id"}, {ArtifactUnOp::heapFree, "delete id"},
    {ArtifactUnOp::postIncr, "id++"},  {ArtifactUnOp::postDecr, "id--"}};

  for (auto test : test_cases)
  {
    auto op_type = test.first;
    auto ref_output = test.second;
    checkUnaryExpression(op_type, var, ref_output);
  }
}

static void checkBinaryExpression(ArtifactBinOp op, shared_ptr<ArtifactExpr> op1,
                                  shared_ptr<ArtifactExpr> op2, const char *ref_data)
{
  stringstream code_out;
  ArtifactGeneratorCppCode code_gen(code_out);

  shared_ptr<ArtifactBinaryExpr> expr = AF::bin(op, op1, op2);
  expr->accept(&code_gen);
  ASSERT_EQ(code_out.str(), ref_data);
}

TEST(acl_backend_dom_to_text, ArtifactBinaryExpr)
{
  stringstream code_out;
  ArtifactGeneratorCppCode code_gen(code_out);
  const char *op1_name = "a";
  const char *op2_name = "b";
  shared_ptr<ArtifactId> op1 = AF::id(op1_name);
  shared_ptr<ArtifactId> op2 = AF::id(op2_name);

  pair<ArtifactBinOp, const char *> test_cases[] = {
    {ArtifactBinOp::eq, "a == b"},          {ArtifactBinOp::notEq, "a != b"},
    {ArtifactBinOp::less, "a < b"},         {ArtifactBinOp::lessOrEq, "a <= b"},
    {ArtifactBinOp::great, "a > b"},        {ArtifactBinOp::greatOrEq, "a >= b"},
    {ArtifactBinOp::assign, "a = b"},       {ArtifactBinOp::plus, "a + b"},
    {ArtifactBinOp::minus, "a - b"},        {ArtifactBinOp::mult, "a * b"},
    {ArtifactBinOp::div, "a / b"},          {ArtifactBinOp::plusAssign, "a += b"},
    {ArtifactBinOp::minusAssign, "a -= b"}, {ArtifactBinOp::multAssign, "a *= b"},
    {ArtifactBinOp::divAssign, "a /= b"}};

  for (auto test : test_cases)
  {
    auto op_type = test.first;
    auto ref_output = test.second;
    checkBinaryExpression(op_type, op1, op2, ref_output);
  }
}

TEST(acl_backend_dom_to_text, ArtifactIndex)
{
  stringstream code_out;
  ArtifactGeneratorCppCode code_gen(code_out);
  const char *arr_name = "a";
  const char *idx_name = "b";
  shared_ptr<ArtifactId> arr = AF::id(arr_name);
  shared_ptr<ArtifactId> idx = AF::id(idx_name);
  shared_ptr<ArtifactIndex> indexing = AF::ind(arr, idx);
  indexing->accept(&code_gen);
  ASSERT_EQ(code_out.str(), "a[b]");
}

TEST(acl_backend_dom_to_text, ArtifactRet)
{
  stringstream code_out;
  ArtifactGeneratorCppCode code_gen(code_out);
  const char *result_name = "a";
  shared_ptr<ArtifactId> result = AF::id(result_name);
  ArtifactRet ret(result);
  ret.accept(&code_gen);
  ASSERT_EQ(code_out.str(), "return a");
}

TEST(acl_backend_dom_to_text, ArtifactBreak)
{
  stringstream code_out;
  ArtifactGeneratorCppCode code_gen(code_out);
  ArtifactBreak brk;
  brk.accept(&code_gen);
  ASSERT_EQ(code_out.str(), "break");
}

TEST(acl_backend_dom_to_text, ArtifactCont)
{
  stringstream code_out;
  ArtifactGeneratorCppCode code_gen(code_out);
  ArtifactCont cont;
  cont.accept(&code_gen);
  ASSERT_EQ(code_out.str(), "continue");
}

TEST(acl_backend_dom_to_text, ArtifactVariable)
{
  stringstream code_out;
  ArtifactGeneratorCppCode code_gen(code_out);
  const char *var_type = "int";
  const char *var_name = "data";
  shared_ptr<ArtifactLiteral> dim1 = AF::lit("2");
  shared_ptr<ArtifactLiteral> dim2 = AF::lit("3");
  list<shared_ptr<ArtifactExpr>> dims{dim1, dim2};
  list<shared_ptr<ArtifactExpr>> initializers{AF::lit("123")};
  shared_ptr<ArtifactVariable> var_decl = AF::var(var_type, var_name, dims, initializers);
  var_decl->accept(&code_gen);
  // TODO generate initializers in braces
  ASSERT_EQ(code_out.str(), "int data[2][3](123)");
}

TEST(acl_backend_dom_to_text, ArtifactBlock)
{
  stringstream code_out;
  ArtifactGeneratorCppCode code_gen(code_out);
  const char *var_name = "var";
  const char *lit_val = "123";

  shared_ptr<ArtifactExpr> id = AF::id(var_name);
  shared_ptr<ArtifactExpr> lit = AF::lit(lit_val);
  const list<shared_ptr<ArtifactExpr>> args{id, lit};

  shared_ptr<ArtifactFunctionCall> call = AF::call("foo", args);

  ArtifactBlock block;

  block.addStatement(call);

  block.accept(&code_gen);
  ASSERT_EQ(code_out.str(), " {\n  foo(var, 123);\n}\n");
}

TEST(acl_backend_dom_to_text, ArtifactForLoop)
{
  stringstream code_out;
  ArtifactGeneratorCppCode code_gen(code_out);
  const char *var_name = "i";
  const char *var_type = "int";

  shared_ptr<ArtifactVariable> iter = AF::var(var_type, var_name, {}, {AF::lit("0")});
  shared_ptr<ArtifactExpr> step =
    AF::bin(ArtifactBinOp::plusAssign, AF::id(var_name), AF::lit("1"));
  shared_ptr<ArtifactExpr> cond =
    AF::bin(ArtifactBinOp::lessOrEq, AF::id(var_name), AF::lit("123"));

  shared_ptr<ArtifactBinaryExpr> expr =
    AF::bin(ArtifactBinOp::plusAssign, AF::id("hello"), AF::id("world"));

  ArtifactForLoop loop(iter, cond, step);

  loop.getBlock()->addStatement(expr);

  loop.accept(&code_gen);
  ASSERT_EQ(code_out.str(), "for(int i(0); i <= 123; i += 1) {\n  hello += world;\n}\n");
}

TEST(acl_backend_dom_to_text, ArtifactIf)
{
  stringstream code_out;
  ArtifactGeneratorCppCode code_gen(code_out);
  const char *var_name = "i";

  shared_ptr<ArtifactExpr> cond =
    AF::bin(ArtifactBinOp::lessOrEq, AF::id(var_name), AF::lit("123"));

  shared_ptr<ArtifactBinaryExpr> expr =
    AF::bin(ArtifactBinOp::plusAssign, AF::id("hello"), AF::id("world"));

  ArtifactIf if_stmt(cond);

  if_stmt.getBlock()->addStatement(expr);

  if_stmt.accept(&code_gen);
  ASSERT_EQ(code_out.str(), "if(i <= 123) {\n  hello += world;\n}\n");
}

TEST(acl_backend_dom_to_text, ArtifactFunction)
{
  stringstream code_out;
  stringstream decl_out;
  ArtifactGeneratorCppCode code_gen(code_out);
  ArtifactGeneratorCppDecl decl_gen(decl_out);
  const char *ret_type = "int";
  const char *func_name = "foo";
  shared_ptr<ArtifactVariable> arg1 = AF::var("int", "a");
  shared_ptr<ArtifactVariable> arg2 = AF::var("bool", "b");
  list<shared_ptr<ArtifactVariable>> args{arg1, arg2};

  // test public class variable
  ArtifactFunction func_decl(ret_type, func_name, args);

  func_decl.accept(&code_gen);
  ASSERT_EQ(code_out.str(), "");
  func_decl.accept(&decl_gen);

  ASSERT_EQ(decl_out.str(), "int foo(int a, bool b);");
}

TEST(acl_backend_dom_to_text, ArtifactClassVariable)
{
  stringstream code_out;
  stringstream decl_out;
  ArtifactGeneratorCppCode code_gen(code_out);
  ArtifactGeneratorCppDecl decl_gen(decl_out);

  const char *var_type = "int";
  const char *var_name = "data";

  ArtifactClass cls("Class");

  shared_ptr<ArtifactLiteral> dim1 = AF::lit("2");
  shared_ptr<ArtifactLiteral> dim2 = AF::lit("3");
  list<shared_ptr<ArtifactExpr>> dims{dim1, dim2};
  list<shared_ptr<ArtifactExpr>> list_of_initializer{AF::lit("123")};
  ArtifactClassVariable var_decl(&cls, var_type, var_name, dims, list_of_initializer);

  var_decl.accept(&code_gen);
  ASSERT_EQ(code_out.str(), "data(123)");
  var_decl.accept(&decl_gen);
  // fixme dimensions are not taken into account, remove ';'
  ASSERT_EQ(decl_out.str(), "int data;\n");
}

TEST(acl_backend_dom_to_text, ArtifactClassFunction)
{
  stringstream code_out;
  stringstream decl_out;
  ArtifactGeneratorCppCode code_gen(code_out);
  ArtifactGeneratorCppDecl decl_gen(decl_out);
  const char *ret_type = "int";
  const char *func_name = "foo";
  shared_ptr<ArtifactVariable> arg1 = AF::var("int", "a");
  shared_ptr<ArtifactVariable> arg2 = AF::var("bool", "b");
  list<shared_ptr<ArtifactVariable>> args{arg1, arg2};

  ArtifactClass cls("Class");

  // test public class variable
  shared_ptr<ArtifactClassFunction> cls_func_decl = cls.func(true, ret_type, func_name, args);

  cls_func_decl->accept(&code_gen);
  // FIXME do not print new line in this visitor
  ASSERT_EQ(code_out.str(), "int Class::foo(int a, bool b) {\n}\n\n");
  cls_func_decl->accept(&decl_gen);

  ASSERT_EQ(decl_out.str(), "int foo(int a, bool b);\n");

  decl_out.str("");
  code_out.str("");

  // test private class variable
  cls_func_decl = cls.func(false, ret_type, func_name, args);

  cls_func_decl->accept(&code_gen);
  // FIXME do not print new line in this visitor
  ASSERT_EQ(code_out.str(), "int Class::foo(int a, bool b) {\n}\n\n");
  cls_func_decl->accept(&decl_gen);

  ASSERT_EQ(decl_out.str(), "int foo(int a, bool b);\n");
}

static shared_ptr<ArtifactClassVariable> createClsVariable(ArtifactClass &cls, const char *var_name,
                                                           bool is_public)
{
  const char *var_type = "int";
  shared_ptr<ArtifactLiteral> dim1 = AF::lit("2");
  shared_ptr<ArtifactLiteral> dim2 = AF::lit("3");
  list<shared_ptr<ArtifactExpr>> dims{dim1, dim2};
  list<shared_ptr<ArtifactExpr>> initializers{AF::lit("123")};
  shared_ptr<ArtifactClassVariable> var_decl =
    cls.var(is_public, var_type, var_name, dims, initializers);
  return var_decl;
}

static shared_ptr<ArtifactClassFunction> createClsFunction(ArtifactClass &cls,
                                                           const char *func_name, bool is_public)
{
  const char *var_type = "int";
  const char *func_type = "void";
  shared_ptr<ArtifactVariable> var1 = AF::var(var_type, "a");
  shared_ptr<ArtifactVariable> var2 = AF::var(var_type, "b");
  list<shared_ptr<ArtifactVariable>> args{var1, var2};
  shared_ptr<ArtifactClassFunction> func_decl = cls.func(is_public, func_type, func_name, args);
  return func_decl;
}

TEST(acl_backend_dom_to_text, ArtifactClass)
{
  stringstream code_out;
  stringstream decl_out;
  ArtifactGeneratorCppCode code_gen(code_out);
  ArtifactGeneratorCppDecl decl_gen(decl_out);

  ArtifactClass cls("Class");

  createClsFunction(cls, "public_foo", true);
  createClsFunction(cls, "private_bar", false);

  createClsVariable(cls, "visible", true);
  createClsVariable(cls, "invisible", false);

  // Test cpp file generation
  cls.accept(&code_gen);
  ASSERT_EQ(code_out.str(), "Class::Class() : invisible(123) {\n}\n\n"
                            "void Class::public_foo(int a, int b) {\n}\n\n"
                            "void Class::private_bar(int a, int b) {\n}\n\n");

  // Test header file generation
  cls.accept(&decl_gen);

  ASSERT_EQ(decl_out.str(), "class Class {\npublic:\n  Class();\n"
                            "  void public_foo(int a, int b);"
                            "\n\nprivate:\n  void private_bar(int a, int b);\n\n"
                            "  int invisible;\n};\n");
}

TEST(acl_backend_dom_to_text, ArtifactModule)
{
  stringstream code_out;
  stringstream decl_out;
  ArtifactGeneratorCppCode code_gen(code_out);
  ArtifactGeneratorCppDecl decl_gen(decl_out);

  ArtifactModule m("module");

  m.addHeaderInclude("foo.h");
  m.addHeaderSysInclude("vector");
  m.addSourceInclude("bar.h");
  m.addSourceSysInclude("list");

  shared_ptr<ArtifactClass> cls = m.createClass("Class");

  // test cpp file generation
  // We use snippet code to encode some common functions
  // This snippet is wrapped in prefix and postfix code
  const char *code_prefix = "#include \"module.h\"\n\n#include <list>\n\n#include \"bar.h\"\n\n";
  const char *code_suffix = "\nClass::Class() {\n}\n\n";

  string ref_data =
    string(code_prefix) + string(AclArtifactUtilities, sizeof(AclArtifactUtilities)) + code_suffix;
  m.accept(&code_gen);
  ASSERT_EQ(code_out.str(), ref_data);

  // test header code generation
  const char *ref_decl_data = "#include <vector>\n\n#include \"foo.h\"\n\nclass Class {\npublic:\n "
                              " Class();\n\nprivate:\n};\n";
  m.accept(&decl_gen);

  ASSERT_EQ(decl_out.str(), ref_decl_data);
}
