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

/**
 * @file  Debugging.h
 * @brief This file includes various interactive debugging helpers
 */

#ifndef __ENCO_SUPPORT_DEBUGGING_H__
#define __ENCO_SUPPORT_DEBUGGING_H__

#include <coco/IR.h>

static_assert(sizeof(long) == sizeof(void *), "sizeof(long) == sizeof(pointer)");

/**
 * Debugging API with a single pointer argument
 */
#define DEBUGGING_API_P(NAME, TYPE) \
  void NAME(const TYPE *);          \
  void NAME(long);

/**
 * Print the details of all the allocated coco::Bag in coco::Module
 *
 * (gdb) call enco_dump_all_bags(bag->module())
 * (gdb) call enco_dump_all_bags(0x...)
 */
DEBUGGING_API_P(enco_dump_all_bags, coco::Module);

/**
 * Print the details of all the allocated coco::Object in coco::Module
 *
 * (gdb) call enco_dump_all_objects(obj->module())
 * (gdb) call enco_dump_all_objects(0x...)
 */
DEBUGGING_API_P(enco_dump_all_objects, coco::Module);

/**
 * Print the details of coco::Op
 *
 * (gdb) call enco_dump_op(op)
 * (gdb) call enco_dump_op(0x....)
 */
DEBUGGING_API_P(enco_dump_op, coco::Op);

/**
 * Print the (simplified) tree layout of coco::Op
 *
 * (gdb) call enco_dump_op_tree(op)
 * (gdb) call enco_dump_op_tree(0x....)
 */
DEBUGGING_API_P(enco_dump_op_tree, coco::Op);

/**
 * Print the details of all the allocated coco::Op in coco::Module
 *
 * (gdb) call enco_dump_all_ops(op->module())
 * (gdb) call enco_dump_all_ops(0x....)
 */
DEBUGGING_API_P(enco_dump_all_ops, coco::Module);

/**
 * Print the details of all the allocated coco::Instr in coco::Module
 *
 * (gdb) call enco_dump_all_instrs(instr->module())
 * (gdb) call enco_dump_all_instrs(0x...)
 */
DEBUGGING_API_P(enco_dump_all_instrs, coco::Module);

/**
 * Print the more details of all the allocated coco::Instr in coco::Module
 *
 * (gdb) call enco_dump_all_instrs_v(instr->module())
 * (gdb) call enco_dump_all_instrs_v(0x...)
 */
DEBUGGING_API_P(enco_dump_all_instrs_v, coco::Module);

/**
 * Print the details of a given coco::Instr
 *
 * (gdb) call enco_dump_instr(instr)
 * (gdb) call enco_dump_instr(0x...)
 */
DEBUGGING_API_P(enco_dump_instr, coco::Instr);

/**
 * Print the details of all the instruction in a given block
 *
 * (gdb) call enco_dump_block(b)
 * (gdb) call enco_dump_block(0x...)
 */
DEBUGGING_API_P(enco_dump_block, coco::Block);

#undef DEBUGGING_API_P

#endif // __ENCO_SUPPORT_DEBUGGING_H__
