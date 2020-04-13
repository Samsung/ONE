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

#ifndef __RECIPE_CHEF_H__
#define __RECIPE_CHEF_H__

#include <mio/tflite/schema_generated.h>
#include <tflchef.pb.h>

#include <memory>
#include <string>

namespace tflchef
{

/**
 * @brief Create ModelRecipe from tflite::Model
 */
std::unique_ptr<ModelRecipe> generate_recipe(const tflite::Model *model);

/**
 * @brief Write ModelRecipe to file with given name
 */
bool write_recipe(const std::string &filename, std::unique_ptr<ModelRecipe> &recipe);

} // namespace tflchef

#endif // __RECIPE_CHEF_H__
