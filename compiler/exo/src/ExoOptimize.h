/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __OPTIMIZE_H__
#define __OPTIMIZE_H__

#include <loco.h>

namespace exo
{

/**
 * @brief Run passes for a graph after completion of converting canonical nodes into TFL nodes.
 *
 * TODO Separate optimize pass dedicated to TFL and Circle dialect when necessary
 */
void optimize(loco::Graph *);

} // namespace exo

#endif // __OPTIMIZE_H__
