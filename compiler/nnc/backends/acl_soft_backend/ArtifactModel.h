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

#ifndef _NNC_ARTIFACT_MODEL_H_
#define _NNC_ARTIFACT_MODEL_H_

#include <string>
#include <list>
#include <ostream>
#include <utility>
#include <memory>

#include "IArtifactGenerator.h"

namespace nnc
{

/**
 * @todo FIXME: remove the identical accept() function implementations!
 *
 * @todo Get rid of the 'Artifact' prefix in the class names
 * in this hierarchy, after anticipated namespace refactoring
 * in the nnc project.
 */

/**
 * @brief Indicates how object on which call is made is treated.
 * In C++ syntax syntax: '.', '->', '::'
 */
enum class ArtifactCallType
{
  obj,  // '.'
  ref,  // '->'
  scope // '::'
};

/**
 * @brief The base class of the whole artifact entities hierarchy.
 */
class ArtifactEntity
{
public:
  virtual ~ArtifactEntity() = default;

  /**
   * @brief If this entity represents something containing a block of instructions
   */
  virtual bool isBlock() const { return false; }
  /**
   * @brief This is the core function of each artifact entity and
   * is implemented by all concrete classes in the hierarchy.
   */
  virtual void accept(IArtifactGenerator *g) const = 0;

protected:
  ArtifactEntity() = default;
};

/**
 * @brief Represents any named entity in the code.
 */
class ArtifactNamed : public ArtifactEntity
{
public:
  explicit ArtifactNamed(std::string name) : _name(std::move(name)) {}

  /**
   * Returns the identifier name.
   * @return the identifier name.
   */
  const std::string &name() const { return _name; }

private:
  std::string _name;
};

/**
 * @brief Basic class for all expressions: identifiers, function calls, references etc.
 */
class ArtifactExpr : public ArtifactEntity
{
};

/**
 * @brief Represents literals which should go to the artifact source code as is.
 */
class ArtifactLiteral : public ArtifactExpr
{
public:
  explicit ArtifactLiteral(std::string value) : _value(std::move(value)) {}

  void accept(IArtifactGenerator *g) const override { g->visit(this); }

  /**
   * Returns the literal value.
   * @return the literal value.
   */
  const std::string &getValue() const { return _value; }

private:
  std::string _value;
};

/**
 * @brief Type of objects which can be used to reference named entities by their names.
 */
class ArtifactId : public ArtifactExpr
{
public:
  explicit ArtifactId(std::string id) : _id(std::move(id)) {}
  explicit ArtifactId(const ArtifactNamed *named) : _id(named->name()) {}

  void accept(IArtifactGenerator *g) const override { g->visit(this); }

  const std::string &name() const { return _id; }

private:
  std::string _id;
};

/**
 * @brief Represents an entity with semantics like C/C++ address of (&) operator.
 */
class ArtifactRef : public ArtifactExpr
{
public:
  explicit ArtifactRef(std::shared_ptr<ArtifactExpr> ref) : _ref(std::move(ref)) {}

  void accept(IArtifactGenerator *g) const override { g->visit(this); }

  std::shared_ptr<ArtifactExpr> obj() const { return _ref; }

private:
  std::shared_ptr<ArtifactExpr> _ref;
};

/**
 * @brief Represents an entity with semantics of C/C++ dereference (*) operator.
 */
class ArtifactDeref : public ArtifactExpr
{
public:
  explicit ArtifactDeref(std::shared_ptr<ArtifactExpr> ref) : _ref(std::move(ref)) {}

  void accept(IArtifactGenerator *g) const override { g->visit(this); }

  std::shared_ptr<ArtifactExpr> obj() const { return _ref; }

private:
  std::shared_ptr<ArtifactExpr> _ref;
};

/**
 * @brief Represents a function call.
 */
class ArtifactFunctionCall : public ArtifactExpr
{
public:
  ArtifactFunctionCall(std::string func_name, std::list<std::shared_ptr<ArtifactExpr>> param_list,
                       std::shared_ptr<ArtifactExpr> on = nullptr,
                       ArtifactCallType call_type = ArtifactCallType::obj);

  void accept(IArtifactGenerator *g) const override { g->visit(this); }

  const std::string &funcName() const { return _funcName; }
  const std::list<std::shared_ptr<ArtifactExpr>> &paramList() const { return _paramList; }
  std::shared_ptr<ArtifactExpr> on() const { return _on; }
  ArtifactCallType callType() const { return _callType; }

private:
  std::string _funcName;
  ArtifactCallType _callType;
  std::shared_ptr<ArtifactExpr> _on;
  std::list<std::shared_ptr<ArtifactExpr>> _paramList;
};

/**
 * @brief Used for the ArtifactUnaryExpr.
 */
enum class ArtifactUnOp
{
  preIncr,
  preDecr,
  heapNew,
  heapFree,
  postIncr,
  postDecr
};

class ArtifactUnaryExpr : public ArtifactExpr
{
public:
  ArtifactUnaryExpr(ArtifactUnOp op, std::shared_ptr<ArtifactExpr> expr)
    : _op(op), _expr(std::move(expr))
  {
  }

  void accept(IArtifactGenerator *g) const override { g->visit(this); }

  ArtifactUnOp getOp() const { return _op; }
  std::shared_ptr<ArtifactExpr> getExpr() const { return _expr; }

private:
  ArtifactUnOp _op;
  std::shared_ptr<ArtifactExpr> _expr;
};

/**
 * @brief Used for the ArtifactBinaryExpr.
 */
enum class ArtifactBinOp
{
  eq,
  notEq,
  less,
  lessOrEq,
  great,
  greatOrEq,
  assign,
  plus,
  minus,
  mult,
  div,
  plusAssign,
  minusAssign,
  multAssign,
  divAssign
};

/**
 * @brief Represents different types of binary expressions.
 */
class ArtifactBinaryExpr : public ArtifactExpr
{
public:
  ArtifactBinaryExpr(ArtifactBinOp op, std::shared_ptr<ArtifactExpr> left,
                     std::shared_ptr<ArtifactExpr> right)
    : _op(op), _left(std::move(left)), _right(std::move(right))
  {
  }

  void accept(IArtifactGenerator *g) const override { g->visit(this); }

  ArtifactBinOp getOp() const { return _op; }
  std::shared_ptr<ArtifactExpr> getLeft() const { return _left; }
  std::shared_ptr<ArtifactExpr> getRight() const { return _right; }

private:
  ArtifactBinOp _op;
  std::shared_ptr<ArtifactExpr> _left;
  std::shared_ptr<ArtifactExpr> _right;
};

/**
 * @brief Array index access
 */
class ArtifactIndex : public ArtifactExpr
{
public:
  ArtifactIndex(std::shared_ptr<ArtifactExpr> expr, std::shared_ptr<ArtifactExpr> ind)
    : _expr(std::move(expr)), _ind(std::move(ind))
  {
  }

  void accept(IArtifactGenerator *g) const override { g->visit(this); }

  std::shared_ptr<ArtifactExpr> getExpr() const { return _expr; }
  std::shared_ptr<ArtifactExpr> getInd() const { return _ind; }

private:
  std::shared_ptr<ArtifactExpr> _expr;
  std::shared_ptr<ArtifactExpr> _ind;
};

/**
 * @brief Just represents return from function statement.
 */
class ArtifactRet : public ArtifactEntity
{
public:
  explicit ArtifactRet(std::shared_ptr<ArtifactExpr> expr) : _expr(std::move(expr)) {}

  void accept(IArtifactGenerator *g) const override { g->visit(this); }

  std::shared_ptr<ArtifactExpr> expr() const { return _expr; }

private:
  std::shared_ptr<ArtifactExpr> _expr;
};

/**
 * @brief Just represents the break statement.
 */
class ArtifactBreak : public ArtifactEntity
{
public:
  void accept(IArtifactGenerator *g) const override { g->visit(this); }
};

/**
 * @brief Just represents the continue statement.
 */
class ArtifactCont : public ArtifactEntity
{
public:
  void accept(IArtifactGenerator *g) const override { g->visit(this); }
};

/**
 * @brief Represents a variable.
 */
class ArtifactVariable : public ArtifactNamed
{
public:
  ArtifactVariable(std::string type_name, std::string var_name,
                   std::list<std::shared_ptr<ArtifactExpr>> dimensions = {},
                   std::list<std::shared_ptr<ArtifactExpr>> initializers = {})
    : _typeName(std::move(type_name)), _dimensions(std::move(dimensions)),
      _initializers(std::move(initializers)), ArtifactNamed(std::move(var_name))
  {
  }

  void accept(IArtifactGenerator *g) const override { g->visit(this); }

  const std::string &typeName() const { return _typeName; }
  const std::list<std::shared_ptr<ArtifactExpr>> &getDimensions() const { return _dimensions; };
  const std::list<std::shared_ptr<ArtifactExpr>> &getInitializers() const { return _initializers; }
  std::shared_ptr<ArtifactId> use() { return std::make_shared<ArtifactId>(this); }

private:
  std::string _typeName;
  std::list<std::shared_ptr<ArtifactExpr>> _dimensions; // If not empty, this is an array
  std::list<std::shared_ptr<ArtifactExpr>> _initializers;
};

/**
 * @brief Represents a block of instructions.
 */
class ArtifactBlock : public ArtifactEntity
{
public:
  bool isBlock() const override { return true; }

  void accept(IArtifactGenerator *g) const override { g->visit(this); }

  void addStatement(const std::shared_ptr<ArtifactEntity> &statement)
  {
    _statements.push_back(statement);
  }

  const std::list<std::shared_ptr<ArtifactEntity>> &getStatements() const { return _statements; }

  /**
   * @brief Creates a new variable and place it to the block.
   * @param type_name - the variable type name.
   * @param var_name - the varibale name.
   * @param dimensions - optional dimensions, if the declared variable is an array.
   * @param initializers - optional arguments of the object constructor.
   * @return - the newly created variable.
   */
  std::shared_ptr<ArtifactVariable>
  var(const std::string &type_name, const std::string &var_name,
      const std::list<std::shared_ptr<ArtifactExpr>> &dimensions = {},
      const std::list<std::shared_ptr<ArtifactExpr>> &initializers = {});
  /**
   * @brief Creates a function call.
   * @param func_name - the function name.
   * @param param_list - the parameters which are used for the call.
   * @param call_on - optional object on which the function is called (if it is a member function).
   * @param call_type - (for member functions only) call through: '.', '->', or '::'.
   * @return
   */
  std::shared_ptr<ArtifactFunctionCall>
  call(const std::string &func_name, const std::list<std::shared_ptr<ArtifactExpr>> &param_list,
       std::shared_ptr<ArtifactExpr> call_on = nullptr,
       ArtifactCallType call_type = ArtifactCallType::obj);
  /**
   * @brief Creates a return from function statement.
   * @param expr - value to return in generated code.
   * @return
   */
  std::shared_ptr<ArtifactRet> ret(std::shared_ptr<ArtifactExpr> expr);

  /**
   * @brief Creates a break from a loop instruction.
   * @return
   */
  std::shared_ptr<ArtifactBreak> brk();

  /**
   * @brief Creates a continue in a loop instruction.
   * @return
   */
  std::shared_ptr<ArtifactCont> cont();

  /**
   * @brief Creates a for loop instruction.
   * @param init - initialize for loop.
   * @param cond - condition to check for stay looping.
   * @param iter - change when transiting to the next iteration.
   * @return
   */
  std::shared_ptr<ArtifactForLoop> forLoop(std::shared_ptr<ArtifactVariable> init = nullptr,
                                           std::shared_ptr<ArtifactExpr> cond = nullptr,
                                           std::shared_ptr<ArtifactExpr> iter = nullptr);

  /**
   * @brief Creates an 'if' blocks.
   * @param cond - condition expression
   * @return
   */
  std::shared_ptr<ArtifactIf> ifCond(std::shared_ptr<ArtifactExpr> cond);

  /**
   * @brief Creates an unary operation expression.
   * @param op
   * @param expr
   * @return
   */
  std::shared_ptr<ArtifactUnaryExpr> un(ArtifactUnOp op, std::shared_ptr<ArtifactExpr> expr);

  /**
   * @brief Creates a binary operation expression.
   * @param op
   * @param left
   * @param right
   * @return
   */
  std::shared_ptr<ArtifactBinaryExpr> bin(ArtifactBinOp op, std::shared_ptr<ArtifactExpr> left,
                                          std::shared_ptr<ArtifactExpr> right);

  /**
   * @brief Creates a heap new operation expression.
   * @param expr
   * @return
   */
  std::shared_ptr<ArtifactUnaryExpr> heapNew(std::shared_ptr<ArtifactExpr> expr);

  /**
   * @brief Creates a heap free operation expression.
   * @param expr
   * @return
   */
  std::shared_ptr<ArtifactUnaryExpr> heapFree(std::shared_ptr<ArtifactExpr> expr);

private:
  std::list<std::shared_ptr<ArtifactEntity>> _statements;
};

/**
 * @brief Represents for loops.
 */
class ArtifactForLoop : public ArtifactEntity
{
public:
  explicit ArtifactForLoop(std::shared_ptr<ArtifactVariable> init = nullptr,
                           std::shared_ptr<ArtifactExpr> cond = nullptr,
                           std::shared_ptr<ArtifactExpr> iter = nullptr)
    : _init(std::move(init)), _cond(std::move(cond)), _iter(std::move(iter))
  {
  }

  bool isBlock() const override { return true; }

  void accept(IArtifactGenerator *g) const override { g->visit(this); }

  std::shared_ptr<ArtifactVariable> getInit() const { return _init; }
  std::shared_ptr<ArtifactExpr> getCond() const { return _cond; }
  std::shared_ptr<ArtifactExpr> getIter() const { return _iter; }
  const ArtifactBlock *getBlock() const { return &_body; }
  ArtifactBlock *getBlock() { return &_body; }

private:
  std::shared_ptr<ArtifactVariable> _init;
  std::shared_ptr<ArtifactExpr> _cond;
  std::shared_ptr<ArtifactExpr> _iter;
  ArtifactBlock _body;
};

/**
 * @brief Represents if block.
 */
class ArtifactIf : public ArtifactEntity
{
public:
  explicit ArtifactIf(std::shared_ptr<ArtifactExpr> cond) : _cond(std::move(cond)) {}
  bool isBlock() const override { return true; }

  void accept(IArtifactGenerator *g) const override { g->visit(this); }

  std::shared_ptr<ArtifactExpr> getCond() const { return _cond; }
  const ArtifactBlock *getBlock() const { return &_body; }
  ArtifactBlock *getBlock() { return &_body; }
  const ArtifactBlock *getElseBlock() const { return &_elseBody; }
  ArtifactBlock *getElseBlock() { return &_elseBody; }

private:
  std::shared_ptr<ArtifactExpr> _cond;
  ArtifactBlock _body;
  ArtifactBlock _elseBody;
};

/**
 * @brief Represents a function.
 */
class ArtifactFunction : public ArtifactNamed
{
public:
  /**
   * @brief Constructs a function object.
   * @param ret_type_name - the name of the returned type
   * @param func_name - the function name.
   * @param params - the parameter list.
   */
  ArtifactFunction(std::string ret_type_name, const std::string &func_name,
                   std::list<std::shared_ptr<ArtifactVariable>> params = {})
    : ArtifactNamed(func_name), _params(std::move(params)), _retTypeName(std::move(ret_type_name))
  {
  }

  void accept(IArtifactGenerator *g) const override { g->visit(this); }

  const std::list<std::shared_ptr<ArtifactVariable>> &getParameters() const { return _params; }
  const std::string &getRetTypeName() const { return _retTypeName; }
  const ArtifactBlock *getBlock() const { return &_body; }
  ArtifactBlock *getBlock() { return &_body; }

private:
  std::list<std::shared_ptr<ArtifactVariable>> _params;
  std::string _retTypeName;
  ArtifactBlock _body;
};

/**
 * @brief Basic class for both class member variables and memmber functions.
 */
class ArtifactClassMember
{
public:
  explicit ArtifactClassMember(const ArtifactClass *owner) : _owner(owner) {}

  const ArtifactClass *owner() const { return _owner; }

protected:
  const ArtifactClass *_owner;
};

/**
 * @brief A class member variables.
 */
class ArtifactClassVariable : public ArtifactClassMember, public ArtifactVariable
{
public:
  ArtifactClassVariable(const ArtifactClass *owner, const std::string &type_name,
                        const std::string &var_name,
                        const std::list<std::shared_ptr<ArtifactExpr>> &dimensions = {},
                        const std::list<std::shared_ptr<ArtifactExpr>> &initializers = {})
    : ArtifactClassMember(owner), ArtifactVariable(type_name, var_name, dimensions, initializers)
  {
  }

  void accept(IArtifactGenerator *g) const override { g->visit(this); }
};

/**
 * @brief A class for member functions.
 */
class ArtifactClassFunction : public ArtifactClassMember, public ArtifactFunction
{
public:
  ArtifactClassFunction(const ArtifactClass *owner, const std::string &ret_type_name,
                        const std::string &func_name,
                        const std::list<std::shared_ptr<ArtifactVariable>> &params = {})
    : ArtifactClassMember(owner), ArtifactFunction(ret_type_name, func_name, params)
  {
  }

  void accept(IArtifactGenerator *g) const override { g->visit(this); }
};

/**
 * @brief Represents a class.
 */
class ArtifactClass : public ArtifactNamed
{
public:
  explicit ArtifactClass(const std::string &class_name) : ArtifactNamed(class_name) {}

  void accept(IArtifactGenerator *g) const override { g->visit(this); }

  /**
   * @brief Creates a class member variable.
   * @param is_public - if the created variable be public.
   * @param type_name
   * @param var_name
   * @param dimensions
   * @param initializer
   * @param constructors
   * @return
   */
  std::shared_ptr<ArtifactClassVariable>
  var(bool is_public, const std::string &type_name, const std::string &var_name,
      const std::list<std::shared_ptr<ArtifactExpr>> &dimensions = {},
      const std::list<std::shared_ptr<ArtifactExpr>> &initializers = {})
  {
    if (is_public)
    {
      auto var = std::make_shared<ArtifactClassVariable>(this, type_name, var_name, dimensions,
                                                         initializers);
      _publicVariables.push_back(var);
      return var;
    }
    else
    {
      auto var = std::make_shared<ArtifactClassVariable>(this, type_name, var_name, dimensions,
                                                         initializers);
      _privateVariables.push_back(var);
      return var;
    }
  }

  /**
   * @brief Creates a class member function.
   * @param is_public - if the created function be public.
   * @param ret_type_name
   * @param func_name
   * @param params
   * @return
   */
  std::shared_ptr<ArtifactClassFunction>
  func(bool is_public, const std::string &ret_type_name, const std::string &func_name,
       const std::list<std::shared_ptr<ArtifactVariable>> &params = {})
  {
    if (is_public)
    {
      auto func = std::make_shared<ArtifactClassFunction>(this, ret_type_name, func_name, params);
      _publicFunctions.push_back(func);
      return func;
    }
    else
    {
      auto func = std::make_shared<ArtifactClassFunction>(this, ret_type_name, func_name, params);
      _privateFunctions.push_back(func);
      return func;
    }
  }

  const std::list<std::shared_ptr<ArtifactClassVariable>> &publicVariables() const
  {
    return _publicVariables;
  }

  const std::list<std::shared_ptr<ArtifactClassVariable>> &privateVariables() const
  {
    return _privateVariables;
  }

  const std::list<std::shared_ptr<ArtifactClassFunction>> &publicFunctions() const
  {
    return _publicFunctions;
  }

  const std::list<std::shared_ptr<ArtifactClassFunction>> &privateFunctions() const
  {
    return _privateFunctions;
  }

  ArtifactBlock *getConstrBlock() { return &_constrBlock; }

  const ArtifactBlock *getConstrBlock() const { return &_constrBlock; }

private:
  std::list<std::shared_ptr<ArtifactClassVariable>> _publicVariables;
  std::list<std::shared_ptr<ArtifactClassVariable>> _privateVariables;
  std::list<std::shared_ptr<ArtifactClassFunction>> _publicFunctions;
  std::list<std::shared_ptr<ArtifactClassFunction>> _privateFunctions;
  ArtifactBlock _constrBlock;
};

/**
 * @brief Class representing a module in the ACL C++ soft backend.
 */
class ArtifactModule
{
public:
  explicit ArtifactModule(std::string name) : _name(std::move(name)) {}

  void accept(IArtifactGenerator *g) const { g->visit(this); }

  std::shared_ptr<ArtifactClass> createClass(const std::string &name)
  {
    auto a_class = std::make_shared<ArtifactClass>(name);
    _entities.emplace_back(a_class);
    return a_class;
  }

  void addHeaderInclude(const std::string &name) { _headerIncludes.push_back(name); }
  void addSourceInclude(const std::string &name) { _sourceIncludes.push_back(name); }
  void addHeaderSysInclude(const std::string &name) { _headerSysIncludes.push_back(name); }
  void addSourceSysInclude(const std::string &name) { _sourceSysIncludes.push_back(name); }

  const std::string &name() const { return _name; }
  const std::list<std::shared_ptr<ArtifactEntity>> entities() const { return _entities; }
  const std::list<std::string> &headerIncludes() const { return _headerIncludes; }
  const std::list<std::string> &sourceIncludes() const { return _sourceIncludes; }
  const std::list<std::string> &headerSysIncludes() const { return _headerSysIncludes; }
  const std::list<std::string> &sourceSysIncludes() const { return _sourceSysIncludes; }

private:
  std::string _name;
  std::list<std::shared_ptr<ArtifactEntity>> _entities;
  std::list<std::string> _headerIncludes;
  std::list<std::string> _sourceIncludes;
  std::list<std::string> _headerSysIncludes;
  std::list<std::string> _sourceSysIncludes;
};

/**
 * @brief Factory for some kinds of frequently used artifact DOM objects.
 */
class ArtifactFactory
{
public:
  static std::shared_ptr<ArtifactId> id(const std::string &name)
  {
    return std::make_shared<ArtifactId>(name);
  }

  static std::shared_ptr<ArtifactLiteral> lit(const std::string &name)
  {
    return std::make_shared<ArtifactLiteral>(name);
  }

  /**
   * @brief Creates a new variable and place it to the block.
   * @param type_name - the variable type name.
   * @param var_name - the varibale name.
   * @param dimensions - optional dimensions, if the declared variable is an array.
   * @param initializer - optional variable initializer.
   * @param constructors - optional arguments of the object constructor, if instantiating a class.
   * @return - the newly created variable.
   */
  static std::shared_ptr<ArtifactVariable>
  var(const std::string &type_name, const std::string &var_name,
      const std::list<std::shared_ptr<ArtifactExpr>> &dimensions = {},
      const std::list<std::shared_ptr<ArtifactExpr>> &initializers = {})
  {
    return std::make_shared<ArtifactVariable>(type_name, var_name, dimensions, initializers);
  }

  /**
   * @brief Creates a 'reference' (like C/C++ '&' address operator do) to the expression.
   * @param ref
   * @return
   */
  static std::shared_ptr<ArtifactRef> ref(std::shared_ptr<ArtifactExpr> ref)
  {
    return std::make_shared<ArtifactRef>(ref);
  }

  /**
   * @brief Creates a 'dereference' (like C/C++ '*' dereference operator do) of the expression.
   * @param ref
   * @return
   */
  static std::shared_ptr<ArtifactDeref> deref(std::shared_ptr<ArtifactExpr> ref)
  {
    return std::make_shared<ArtifactDeref>(ref);
  }

  /**
   * @brief Creates a function call.
   * @param func_name - the function name.
   * @param param_list - the parameters which are used for the call.
   * @param call_on - optional object on which the function is called (if it is a member function).
   * @param call_type - (for member functions only) call through: '.', '->', or '::'.
   * @return
   */
  static std::shared_ptr<ArtifactFunctionCall>
  call(const std::string &func_name, const std::list<std::shared_ptr<ArtifactExpr>> &param_list,
       std::shared_ptr<ArtifactExpr> on = nullptr,
       ArtifactCallType call_type = ArtifactCallType::obj)
  {
    return std::make_shared<ArtifactFunctionCall>(func_name, param_list, on, call_type);
  }

  /**
   * @brief Creates an unary operation expression.
   * @param op
   * @param expr
   * @return
   */
  static std::shared_ptr<ArtifactUnaryExpr> un(ArtifactUnOp op, std::shared_ptr<ArtifactExpr> expr)
  {
    return std::make_shared<ArtifactUnaryExpr>(op, expr);
  }

  /**
   * @brief Creates a binary operation expression.
   * @param op
   * @param left
   * @param right
   * @return
   */
  static std::shared_ptr<ArtifactBinaryExpr>
  bin(ArtifactBinOp op, std::shared_ptr<ArtifactExpr> left, std::shared_ptr<ArtifactExpr> right)
  {
    return std::make_shared<ArtifactBinaryExpr>(op, left, right);
  }

  /**
   * @brief Creates an array element accessor expression (like C/C++ array[i]).
   * @param expr
   * @param ind
   * @return
   */
  static std::shared_ptr<ArtifactIndex> ind(std::shared_ptr<ArtifactExpr> expr,
                                            std::shared_ptr<ArtifactExpr> ind)
  {
    return std::make_shared<ArtifactIndex>(expr, ind);
  }

  /**
   * @brief Creates a heap new operation expression.
   * @param expr
   * @return
   */
  static std::shared_ptr<ArtifactUnaryExpr> heapNew(std::shared_ptr<ArtifactExpr> expr)
  {
    return std::make_shared<ArtifactUnaryExpr>(ArtifactUnOp::heapNew, expr);
  }

  /**
   * @brief Creates a heap free operation expression.
   * @param expr
   * @return
   */
  static std::shared_ptr<ArtifactUnaryExpr> heapFree(std::shared_ptr<ArtifactExpr> expr)
  {
    return std::make_shared<ArtifactUnaryExpr>(ArtifactUnOp::heapFree, expr);
  }
};

} // namespace nnc

#endif //_NNC_ARTIFACT_MODEL_H_
