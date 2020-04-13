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

#ifndef NNCC_TREE_H
#define NNCC_TREE_H

#include <iostream>
#include <vector>
#include <memory>
#include <random>

#include <stddef.h>

namespace modelgen
{

enum class OpCodes
{
  opConv2d,
  opConcatenation,
  opDepthwiseConv2d,
  opOpMaxPool2d,
  opAveragePool2d,
  opSoftmax,
  opFullyConnected,
  opCount,
  opFirst = 0,
  opLast = opCount - 1,
};

namespace treebuilder
{

/**
 * opcode is a code of operation.
 * inputs is a index of parent operations.
 * inputShape is a shape of left parent.
 * levelOwner is a index of level having the operation.
 *
 * @details In some cases the kernelShape will keep
 *          not the kernel but specific data for operation.
 *
 */
struct Operation
{
  OpCodes opcode;
  std::vector<int32_t> inputs;
  std::vector<int32_t> kernelShape;
  std::vector<int32_t> inputShape;
  std::vector<int32_t> outputShape;
  size_t levelOwner;
};

/**
 * @param inputCnt is a number of tensors to tree's input.
 * @param hTree is a number of levels in tree.
 * @param inputShapeTree is a shape for input tree's tensor.
 * @param widthLevel is a number of operations on the level [i].
 * @param beginLevel keeps the vector of operations id which are beginners for level [i].
 * @param endLevel keeps the vector of operations id which are end for level [i].
 * @param opList keeps the vector of all used operations.
 *
 * @details beginLevel is array of indexes to opList.
 *          for example: beginLevel[4] contains the id of first operation on the level 4.
 *          id is index for opList array.
 */
struct Tree
{
  int inputCnt;
  int hTree;
  std::vector<int32_t> inputShapeTree;
  std::vector<int32_t> widthLevel;
  std::vector<int32_t> beginLevel;
  std::vector<int32_t> endLevel;
  std::vector<std::unique_ptr<Operation>> opList;
};

class TreeBuilder
{
public:
  TreeBuilder();

  std::unique_ptr<Tree> buildTree();
  /**
   * @details initTree creates first level and specific operations on its using inputShapeTree
   *          inside every operation as size of input. This method are used when there aren't
   *          operations on upper levels.
   */
  void initTree(Tree *t);
  void buildLevel(Tree *t);

  bool buildSketchOperation(Tree *t, Operation *op);

  /**
   * @details Currently Conv2D are build with stride = 1.
   * @param input_shape is the shape of input tensor.
   * @param op is the pointer to created operation.
   */
  void buildSketchConv2D(Tree *t, Operation *op);
  void buildSketchConcat(Tree *t, Operation *op);
  void buildSketchDepthwiseConv2D(Tree *t, Operation *op);
  void buildSketchPooling(Tree *t, Operation *op);
  void buildSketchSoftmax(Tree *t, Operation *op);
  void buildSketchFullyConnected(Tree *t, Operation *op);

private:
  /**
   * @lookupConsistentOutput It Looks up operation with conststent output's shape.
   * @param t is a tree where doing search.
   * @param op is a operation for which looking a operation.
   * @param shape is a template of shape. -1 in this vector is ignorid axis.
   *        For example in case of <1, 64, 64, -1> the lookupConsistentOutput looks up
   *        operation with output <1, 64, 64, X>.
   * @param until_level is a level starting from which the searching doesn't perform.
   *
   * @details If the operation doesn't found then this method create such operation
   *          and return its id.
   *
   */
  int32_t lookupConsistentOutput(Tree *t, Operation *op, std::vector<int32_t> &shape,
                                 int32_t until_level);

  void buildSketchConv2DForShape(std::vector<int32_t> &input_shape, Operation *op);
  void addConcatInput(std::vector<int32_t> &input_shape, int32_t axis, Operation *op);

  std::random_device _rd;
  std::mt19937 _gen;
};

} // namespace treebuilder
} // namespace modelgen
#endif // NNCC_TREE_H
