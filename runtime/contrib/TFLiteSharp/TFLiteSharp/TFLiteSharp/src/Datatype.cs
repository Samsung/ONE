/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** Type of elements in a {@link TfLiteTensor}. */
enum DataType
{
    /** 32-bit signed integer. */
    INT32 = 1,

    /** 32-bit single precision floating point. */
    FLOAT32 = 2,

    /** 8-bit unsigned integer. */
    UINT8 = 3,

    /** 64-bit signed integer. */
    INT64 = 4
}
