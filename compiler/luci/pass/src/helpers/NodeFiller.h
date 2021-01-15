/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

namespace luci
{

/**
 * INTRODUCTION
 *         Binary operation f(x,y) is 'commutative' when
 *         f(x,y) == f(y,x) holds for all x, y.
 *         For examples, ADD, MUL and SQUARED_DIFFERENCE are commutative.
 *         These helpers make it easy to find commutative arguments of commutative node.
 *
 * HOW TO USE
 *         COMM_NODE *node;
 *         ARG_TYPE_1 *arg1;
 *         ARG_TYPE_2 *arg2;
 *
 *         bool ok = fill(&arg1, &arg2).with_commutative_args_of(node);
 *
 * Result
 *         If 'node's commutative argument types are actually {ARG_TYPE_1, ARG_TYPE_2}
 *         (as a set), 'arg1' and 'arg2' set as actual 'node's arguments with matching
 *         type, and return value 'ok' is true.
 *         Otherwise, 'arg1' and 'arg2' not changed, 'ok' is false.
 */

template <class ARG_TYPE_1, class ARG_TYPE_2> class NodeFiller final
{
public:
  NodeFiller(ARG_TYPE_1 **arg_1, ARG_TYPE_2 **arg_2) : _arg_1(arg_1), _arg_2(arg_2)
  {
    // DO NOTHING
  }

  /**
   * @return true   When 'node's argument types are 'ARG_TYPE_1' and 'ARG_TYPE_2'
   *                In such case, it assign '_arg_1' and '_arg_2' to actual arguments
   *
   * @return false  When 'node's argument types are NOT matched with 'ARG_TYPE_*'
   *                In such case, it does not amend '_arg_1' and '_arg_2'
   *
   * @require       COMM_NODE has member x() and y()
   */
  template <class COMM_NODE> bool with_commutative_args_of(const COMM_NODE *node);

private:
  ARG_TYPE_1 **_arg_1;
  ARG_TYPE_2 **_arg_2;
};

template <class ARG_TYPE_1, class ARG_TYPE_2>
inline NodeFiller<ARG_TYPE_1, ARG_TYPE_2> fill(ARG_TYPE_1 **arg_1, ARG_TYPE_2 **arg_2)
{
  return NodeFiller<ARG_TYPE_1, ARG_TYPE_2>{arg_1, arg_2};
}

template <class ARG_TYPE_1, class ARG_TYPE_2>
template <class COMM_NODE>
bool NodeFiller<ARG_TYPE_1, ARG_TYPE_2>::with_commutative_args_of(const COMM_NODE *node)
{
  // Case 1) X == ARG_TYPE_1 / Y == ARG_TYPE_2
  {
    auto x = dynamic_cast<ARG_TYPE_1 *>(node->x());
    auto y = dynamic_cast<ARG_TYPE_2 *>(node->y());

    if (x && y)
    {
      *_arg_1 = x;
      *_arg_2 = y;
      return true;
    }
  }

  // Case 2) X == ARG_TYPE_2 / Y == ARG_TYPE_1
  {
    auto x = dynamic_cast<ARG_TYPE_2 *>(node->x());
    auto y = dynamic_cast<ARG_TYPE_1 *>(node->y());

    if (x && y)
    {
      *_arg_1 = y;
      *_arg_2 = x;
      return true;
    }
  }

  return false;
}

} // namespace luci
