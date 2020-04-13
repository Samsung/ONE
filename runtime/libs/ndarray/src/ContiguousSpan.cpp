/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ndarray/ContiguousSpan.h"

namespace ndarray
{

template class ContiguousSpan<float, true>;
template class ContiguousSpan<float, false>;
template class ContiguousSpan<int32_t, true>;
template class ContiguousSpan<int32_t, false>;
template class ContiguousSpan<uint32_t, true>;
template class ContiguousSpan<uint32_t, false>;
template class ContiguousSpan<uint8_t, true>;
template class ContiguousSpan<uint8_t, false>;

} // namespace ndarray
