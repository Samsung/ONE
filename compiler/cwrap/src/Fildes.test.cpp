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

#include "cwrap/Fildes.h"

#include <cstdlib>
#include <string>
#include <stdexcept>

#include <unistd.h>
#include <fcntl.h>

#include <gtest/gtest.h>

#define DECLARE_TEMPLATE(NAME) char NAME[] = "FILDES-TEST-XXXXXX"

namespace
{

int make_temp(char *name_template)
{
  mode_t mask = umask(S_IRWXU);
  int fd = mkstemp(name_template);
  umask(mask);

  if (fd == -1)
  {
    throw std::runtime_error{"mkstemp failed"};
  }

  return fd;
}

} // namespace

TEST(FildesTest, default_constructor)
{
  cwrap::Fildes fildes;

  ASSERT_FALSE(cwrap::valid(fildes));
}

TEST(FildesTest, value_constructor)
{
  DECLARE_TEMPLATE(name_template);

  {
    cwrap::Fildes fildes{make_temp(name_template)};

    ASSERT_TRUE(cwrap::valid(fildes));
  }

  unlink(name_template);
}

TEST(FildesTest, move_constructor)
{
  DECLARE_TEMPLATE(src_template);
  DECLARE_TEMPLATE(dst_template);

  int src_fd = make_temp(src_template);
  int dst_fd = make_temp(dst_template);

  {
    cwrap::Fildes src{src_fd};
    cwrap::Fildes dst{dst_fd};

    dst = std::move(src);

    ASSERT_FALSE(cwrap::valid(src));
    ASSERT_TRUE(cwrap::valid(dst));

    ASSERT_EQ(dst.get(), src_fd);

    // "src_fd" SHOULD be valid, and "dst_fd" SHOULD be closed
    ASSERT_NE(fcntl(src_fd, F_GETFD), -1);
    ASSERT_EQ(fcntl(dst_fd, F_GETFD), -1);
  }

  unlink(src_template);
  unlink(dst_template);
}

TEST(FildesTest, destructor)
{
  DECLARE_TEMPLATE(name_template);

  int fd = make_temp(name_template);

  ASSERT_NE(fcntl(fd, F_GETFD), -1);
  {
    cwrap::Fildes fildes{fd};
  }
  ASSERT_EQ(fcntl(fd, F_GETFD), -1);

  unlink(name_template);
}
