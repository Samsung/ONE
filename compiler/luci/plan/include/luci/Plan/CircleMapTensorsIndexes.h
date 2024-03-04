///*
// * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *    http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//
//#ifndef __LUCI_CIRCLE_MAP_TENSORS_INDEXES_H__
//#define __LUCI_CIRCLE_MAP_TENSORS_INDEXES_H__
//
//#include <luci/IR/CircleNode.h>
//
//#include <utility>
//
//namespace luci
//{
//
//class CircleMapTensorsIndexes
//{
//public:
//  CircleMapTensorsIndexes() = delete;
//
//  CircleMapTensorsIndexes(uint32_t f_i, uint32_t s_i)
//  {
//    _first_idx = f_i;
//    _second_idx = s_i;
//  }
//
//  uint32_t first_idx(void) const { return _first_idx; }
//  void first_idx(const uint32_t &first_idx) { _first_idx = first_idx; }
//
//  uint32_t second_idx(void) const { return _second_idx; }
//  void second_idx(const uint32_t &second_idx) { _second_idx = second_idx; }
//
//private:
//  uint32_t _first_idx = 0;
//  uint32_t _second_idx;
//};
//
//bool has_map_tensors_index(const luci::CircleNode *circle_node);
//
//void add_map_tensors_index(luci::CircleNode *circle_node,
//                        const luci::CircleMapTensorsIndexes &execution_plan);
//
//luci::CircleMapTensorsIndexes get_map_tensors_index(const luci::CircleNode *circle_node);
//
//} // namespace luci
//
//#endif // __LUCI_CIRCLE_NODE_EXECUTION_PLAN_H__
