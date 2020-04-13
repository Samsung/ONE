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

#ifndef __NEST_EXPR_MACRO_H__
#define __NEST_EXPR_MACRO_H__

#ifndef NEST_EXPR_CLASS_NAME
#define NEST_EXPR_CLASS_NAME(Tag) Tag##Node
#endif // NEST_EXPR_CLASS_NAME

#ifndef NEST_EXPR_CAST_METHOD_NAME
#define NEST_EXPR_CAST_METHOD_NAME(Tag) as##Tag
#endif // NEST_EXPR_CAST_METHOD_NAME

#endif // __NEST_EXPR_MACRO_H__
