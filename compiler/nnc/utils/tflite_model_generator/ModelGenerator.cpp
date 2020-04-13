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

#include <iostream>
#include <string>
#include <memory>

#include "Tree.h"

#include "RandomModelBuilder.h"
#include "TFLiteRandomModelBuilder.h"

using namespace modelgen;
using namespace treebuilder;

int main(int argc, const char *argv[])
{
  std::unique_ptr<TreeBuilder> tree_builder(new TreeBuilder);
  auto tree = tree_builder->buildTree();

  std::unique_ptr<RandomModelBuilder> builder(new TFLiteRandomModelBuilder);
  builder->convertTreeToModel(tree.get());

  std::unique_ptr<ModelSaver> saver = builder->createModelSaver();
  saver->saveModel();

  return 0;
}
