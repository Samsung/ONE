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

#include <assert.h>
#include <algorithm>

#include "Tree.h"

namespace modelgen
{
namespace treebuilder
{

static constexpr int levelsMin = 1;
static constexpr int levelsMax = 15;
static constexpr int widthMin = 1;
static constexpr int widthMax = 10;
static constexpr int shapeMin = 1;
static constexpr int shapeMax = 64;
static constexpr int concatCntInputsMin = 4;
static constexpr int concatCntInputsMax = 8;
static constexpr int depthwiseConv2dMultiply = 4;
static constexpr int fullyConnectedMaxWeight = 8;
static constexpr int fullyConnectedKernelDim = 2;

TreeBuilder::TreeBuilder() : _gen(_rd()) {}

std::unique_ptr<Tree> TreeBuilder::buildTree()
{
  std::uniform_int_distribution<int> int_rand(levelsMin, levelsMax);
  auto t = std::unique_ptr<Tree>(new Tree);
  t->inputCnt = 1;
  t->hTree = int_rand(_gen);
  initTree(t.get());

  std::cout << "Build " << t->hTree << " levels in tree" << std::endl;
  for (int i = 1; i < t->hTree; i++)
  {
    buildLevel(t.get());
  }

  std::cout << "operations count " << t->opList.end() - t->opList.begin() << std::endl;
  std::cout << "levels count " << t->widthLevel.end() - t->widthLevel.begin() << std::endl;

  return t;
}

void TreeBuilder::buildLevel(Tree *t)
{
  std::uniform_int_distribution<int32_t> int_rand(widthMin, widthMax);
  auto levelId = t->widthLevel.size();
  auto levelWidth = int_rand(_gen);
  t->widthLevel.push_back(static_cast<int32_t>(levelWidth));
  t->beginLevel.push_back(static_cast<int32_t>(t->opList.size()));
  t->endLevel.push_back(static_cast<int32_t>(t->opList.size() + levelWidth - 1));

  for (int32_t i = 0; i < levelWidth; i++)
  {
    auto op = std::unique_ptr<Operation>(new Operation);
    op->levelOwner = levelId;
    /**
     * If the operation was not created, then repeat the creation.
     */
    if (!buildSketchOperation(t, op.get()))
    {
      i--;
      continue;
    }
    t->opList.push_back(std::move(op));
  }
}

void TreeBuilder::initTree(Tree *t)
{
  std::uniform_int_distribution<int32_t> int_rand(shapeMin, shapeMax);

  std::cout << "Build first level in tree" << std::endl;

  t->widthLevel.push_back(int_rand(_gen));
  t->beginLevel.push_back(0);
  t->endLevel.push_back(0);
  int32_t x = int_rand(_gen) * 2, y = int_rand(_gen) * 2, z = int_rand(_gen) % 8 + 1;
  t->inputShapeTree = {1, x, y, z};

  std::cout << "Initialize first level with width = [ " << t->widthLevel[0] << " ]"
            << " and shape [ 1"
            << " " << x << " " << y << " " << z << " ]" << std::endl;
  for (int32_t i = 0; i < t->widthLevel[0]; i++)
  {
    auto op = std::unique_ptr<Operation>(new Operation);
    op->levelOwner = 0;
    /**
     * If the operation was not created, then repeat the creation.
     */
    if (!buildSketchOperation(t, op.get()))
    {
      i--;
      continue;
    }
    t->opList.push_back(std::move(op));
  }
}

bool TreeBuilder::buildSketchOperation(Tree *t, Operation *op)
{
  std::uniform_int_distribution<int32_t> opcode_rand(static_cast<int32_t>(OpCodes::opFirst),
                                                     static_cast<int32_t>(OpCodes::opLast));

  op->opcode = static_cast<OpCodes>(opcode_rand(_gen));
  switch (op->opcode)
  {
    case OpCodes::opConv2d:
      buildSketchConv2D(t, op);
      break;
    case OpCodes::opConcatenation:
      buildSketchConcat(t, op);
      break;
    case OpCodes::opDepthwiseConv2d:
      buildSketchDepthwiseConv2D(t, op);
      break;
    case OpCodes::opOpMaxPool2d:
    case OpCodes::opAveragePool2d:
      buildSketchPooling(t, op);
      break;
    case OpCodes::opSoftmax:
      buildSketchSoftmax(t, op);
      break;
    case OpCodes::opFullyConnected:
      /**
       * Currently, we can create fullyconnected operation only on last level.
       * @todo fix it.
       */
      if (t->beginLevel.size() != static_cast<size_t>(t->hTree))
      {
        return false;
      }

      buildSketchFullyConnected(t, op);
      break;
    default:
      assert(false && "TreeBuilder: Unsupported operation");
  }

  return true;
}

void TreeBuilder::buildSketchConv2D(Tree *t, Operation *op)
{
  std::uniform_int_distribution<int32_t> int_rand(0, INT32_MAX);

  if (t->beginLevel.size() == 1)
  {
    op->inputShape = t->inputShapeTree;
    buildSketchConv2DForShape(op->inputShape, op);
    return;
  }

  auto levelId = int_rand(_gen) % (t->beginLevel.size() - 1);
  auto opId = t->beginLevel[levelId] + (int_rand(_gen) % t->widthLevel[levelId]);

  std::cout << "input level [ " << levelId << " ] operation id [ " << opId << " ]" << std::endl;

  op->inputs.push_back(opId);
  op->levelOwner = t->beginLevel.size() - 1;
  op->inputShape = t->opList[opId]->outputShape;
  buildSketchConv2DForShape(op->inputShape, op);
}

void TreeBuilder::buildSketchConcat(Tree *t, Operation *op)
{
  std::uniform_int_distribution<int32_t> int_rand(2, INT32_MAX);
  auto axis = 1 + (int_rand(_gen) + 3) % 3;
  auto input_cnt = concatCntInputsMin + int_rand(_gen) % concatCntInputsMax;

  /* Special case if there are only one level (input to neural network) */
  if (t->beginLevel.size() == 1)
  {
    op->inputShape = t->inputShapeTree;
    op->outputShape = op->inputShape;
    for (int i = 0; i < input_cnt; i++)
    {
      op->inputs.push_back(-1); /* -1 means that it is needed to specify amount inputs
                                 * on the first level where input tensor for operation
                                 * is a input tensor for neural network. */
      addConcatInput(op->inputShape, axis, op);
    }
    op->inputShape[axis] = -1; /* specify a dimension for concatenation. */
    return;
  }

  /* Select the first operand */
  auto levelId = int_rand(_gen) % (t->beginLevel.size() - 1);
  auto opId = t->beginLevel[levelId] + (int_rand(_gen) % t->widthLevel[levelId]);
  std::cout << "input level [ " << levelId << " ] operation id [ " << opId << " ]" << std::endl;

  op->inputs.push_back(opId);
  op->levelOwner = t->beginLevel.size() - 1;
  op->inputShape = t->opList[opId]->outputShape;
  op->outputShape = op->inputShape;
  std::vector<int32_t> shape = op->inputShape;
  shape[axis] = -1;
  for (int i = 0; i < input_cnt; i++)
  {
    opId = lookupConsistentOutput(t, op, shape, t->beginLevel.size() - 1);
    op->inputs.push_back(opId);
    addConcatInput(t->opList[opId]->outputShape, axis, op);
  }

  op->inputShape[axis] = -1; /* specify a dimension for concatenation. */
}

void TreeBuilder::buildSketchDepthwiseConv2D(Tree *t, Operation *op)
{
  std::uniform_int_distribution<int32_t> int_rand(1, INT32_MAX);
  /**
   * Currently, on the stage of building arbitrary tree it is enough
   * build DepthwiseConv2D node as Conv2D.
   * @todo: maby there is sense to specifically create OpDepthwiseConv2d.
   */
  buildSketchConv2D(t, op);

  /**
   * Then change the kernel's shape.
   */
  op->kernelShape[0] = int_rand(_gen) % depthwiseConv2dMultiply + 1; /* channel multiplier */
  op->kernelShape[1] = op->inputShape[3];                            /* filter height      */
  op->kernelShape[2] = op->inputShape[2];                            /* filter width       */
  op->kernelShape[3] = op->inputShape[3];                            /* input channels     */
  op->outputShape[3] = op->kernelShape[0] * op->kernelShape[3];
}

void TreeBuilder::buildSketchPooling(Tree *t, Operation *op)
{
  std::uniform_int_distribution<int32_t> int_rand(2, INT32_MAX);

  if (t->beginLevel.size() == 1)
  {
    op->inputShape = t->inputShapeTree;
    op->outputShape = op->inputShape;
    return;
  }

  auto levelId = int_rand(_gen) % (t->beginLevel.size() - 1);
  auto opId = t->beginLevel[levelId] + (int_rand(_gen) % t->widthLevel[levelId]);

  std::cout << "input level [ " << levelId << " ] operation id [ " << opId << " ]" << std::endl;

  op->inputs.push_back(opId);
  op->levelOwner = t->beginLevel.size() - 1;
  op->inputShape = t->opList[opId]->outputShape;
  op->outputShape = op->inputShape;
}

void TreeBuilder::buildSketchSoftmax(Tree *t, Operation *op)
{
  /**
   * We need only select input node, the output shape will be same as input.
   * That is why we use pooling's builder.
   */
  buildSketchPooling(t, op);
}

void TreeBuilder::buildSketchFullyConnected(Tree *t, Operation *op)
{
  std::uniform_int_distribution<int32_t> int_rand(2, fullyConnectedMaxWeight);
  /**
   * 1. Select a input form previous nodes by means of buildSketchPooling
   */
  buildSketchPooling(t, op);

  /**
   * 2. Create a weights for fully connected layer.
   */
  op->kernelShape.resize(fullyConnectedKernelDim);
  op->kernelShape[0] = int_rand(_gen);
  op->kernelShape[1] =
      op->inputShape[0] * op->inputShape[1] * op->inputShape[2] * op->inputShape[3];

  op->outputShape.resize(2);
  op->outputShape[0] = op->kernelShape[0];
  op->outputShape[1] = op->kernelShape[1];
}

// =========== private ===========

int32_t TreeBuilder::lookupConsistentOutput(Tree *t, Operation *op, std::vector<int32_t> &shape,
                                            int32_t until_level)
{
  for (int i = 0, j = 0; i < t->beginLevel[until_level]; i++)
  {
    for (j = 0; j < 4; j++)
    {
      if (shape[j] != t->opList[i]->outputShape[j] && shape[j] != -1)
      {
        j = 0;
        break;
      }
    }
    if (j == 3 && std::find(op->inputs.begin(), op->inputs.end(), i) == op->inputs.end())
      return i;
  }

  /*
   * Help to code below (initialization new_op):
   *       Z = y->inputs
   *      / \
   *     Y   new_op
   *      \ /
   *      op
   */
  const Operation *y = t->opList[op->inputs[0]].get();
  std::unique_ptr<Operation> new_op = std::unique_ptr<Operation>(new Operation(*y));

  /*
   * reindex operations
   */
  auto inser_pos = t->beginLevel[y->levelOwner];
  for (auto &in : op->inputs)
  {
    if (in >= inser_pos)
      in++;
  }
  for (int i = inser_pos; i < static_cast<int32_t>(t->opList.size()); i++)
  {
    for (auto &in : t->opList[i]->inputs)
    {
      if (in >= inser_pos)
      {
        in++;
      }
    }
  }

  t->endLevel[y->levelOwner]++;
  t->widthLevel[y->levelOwner]++;
  for (int i = y->levelOwner + 1; i < static_cast<int32_t>(t->beginLevel.size()); i++)
  {
    t->beginLevel[i]++;
    t->endLevel[i]++;
  }
  t->opList.insert(t->opList.begin() + inser_pos, std::move(new_op));

  return inser_pos;
}

void TreeBuilder::buildSketchConv2DForShape(std::vector<int32_t> &input_shape, Operation *op)
{
  std::uniform_int_distribution<int32_t> int_rand(shapeMin, shapeMax);

  /* out channels */
  op->kernelShape.push_back(input_shape[3]);
  /* filter height */
  op->kernelShape.push_back(int_rand(_gen));
  /* filter width */
  op->kernelShape.push_back(int_rand(_gen));
  /* in channels  */
  op->kernelShape.push_back(input_shape[3]);
  op->outputShape = input_shape;
  std::cout << "\t with kernel [ ";
  for (auto i : op->kernelShape)
    std::cout << " " << i;
  std::cout << " ]" << std::endl;
}

void TreeBuilder::addConcatInput(std::vector<int32_t> &input_shape, int32_t axis, Operation *op)
{
  for (int i = 0; i < 4; i++)
  {
    if (input_shape[i] != op->inputShape[i] && i != axis)
      assert(false && "Not consistency input shapes\n");
  }
  op->outputShape[axis] += input_shape[axis];
}

} // namespace treebuilder
} // namespace modelgen
