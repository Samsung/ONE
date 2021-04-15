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

#include "ConnectNode.h"

namespace
{

void connect(luci::ConnectNode *cn, const luci::CircleNonMaxSuppressionV4 *node)
{
  auto *cloned = loco::must_cast<luci::CircleNonMaxSuppressionV4 *>(cn->find_clone(node));

  luci::CircleNode *boxes = loco::must_cast<luci::CircleNode *>(node->boxes());
  luci::CircleNode *scores = loco::must_cast<luci::CircleNode *>(node->scores());
  luci::CircleNode *max_output_size = loco::must_cast<luci::CircleNode *>(node->max_output_size());
  luci::CircleNode *iou_threshold = loco::must_cast<luci::CircleNode *>(node->iou_threshold());
  luci::CircleNode *score_threshold = loco::must_cast<luci::CircleNode *>(node->score_threshold());

  cloned->boxes(cn->find_clone(boxes));
  cloned->scores(cn->find_clone(scores));
  cloned->max_output_size(cn->find_clone(max_output_size));
  cloned->iou_threshold(cn->find_clone(iou_threshold));
  cloned->score_threshold(cn->find_clone(score_threshold));
}

} // namespace

namespace luci
{

void ConnectNode::visit(const luci::CircleNonMaxSuppressionV4 *node) { connect(this, node); }

} // namespace luci
