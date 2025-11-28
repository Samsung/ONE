/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef _MOCK_SYSCALLS_H_
#define _MOCK_SYSCALLS_H_

#include <sys/mman.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <stdio.h>
#include <functional>
#include <memory>
#include <cstdarg>

namespace onert
{
namespace backend
{
namespace trix
{
namespace ops
{
namespace test
{

class MockSyscallsManager
{
public:
  // Function type definitions for each syscall
  // Note: std::function doesn't work well with variadic functions, so we use specific signatures
  using OpenHook = std::function<int(const char *, int)>;
  using OpenCreatHook = std::function<int(const char *, int, mode_t)>;
  using MmapHook = std::function<void *(void *, size_t, int, int, int, off_t)>;
  using MunmapHook = std::function<int(void *, size_t)>;
  using CloseHook = std::function<int(int)>;
  using IoctlHook = std::function<int(int, unsigned long, void *)>;
  using FopenHook = std::function<FILE *(const char *, const char *)>;
  using FcloseHook = std::function<int(FILE *)>;
  using FreadHook = std::function<size_t(void *, size_t, size_t, FILE *)>;
  using FseekHook = std::function<int(FILE *, long, int)>;

  static MockSyscallsManager &getInstance()
  {
    static MockSyscallsManager instance;
    return instance;
  }

  // Hook registration functions
  void setOpenHook(OpenHook hook) { _openHook = hook; }
  void setOpenCreatHook(OpenCreatHook hook) { _openCreatHook = hook; }
  void setMmapHook(MmapHook hook) { _mmapHook = hook; }
  void setMunmapHook(MunmapHook hook) { _munmapHook = hook; }
  void setCloseHook(CloseHook hook) { _closeHook = hook; }
  void setIoctlHook(IoctlHook hook) { _ioctlHook = hook; }
  void setFopenHook(FopenHook hook) { _fopenHook = hook; }
  void setFcloseHook(FcloseHook hook) { _fcloseHook = hook; }
  void setFreadHook(FreadHook hook) { _freadHook = hook; }
  void setFseekHook(FseekHook hook) { _fseekHook = hook; }

  // Hook retrieval functions
  OpenHook getOpenHook() const { return _openHook; }
  OpenCreatHook getOpenCreatHook() const { return _openCreatHook; }
  MmapHook getMmapHook() const { return _mmapHook; }
  MunmapHook getMunmapHook() const { return _munmapHook; }
  CloseHook getCloseHook() const { return _closeHook; }
  IoctlHook getIoctlHook() const { return _ioctlHook; }
  FopenHook getFopenHook() const { return _fopenHook; }
  FcloseHook getFcloseHook() const { return _fcloseHook; }
  FreadHook getFreadHook() const { return _freadHook; }
  FseekHook getFseekHook() const { return _fseekHook; }

  // Hook clearing functions
  void clearOpenHook() { _openHook = nullptr; }
  void clearOpenCreatHook() { _openCreatHook = nullptr; }
  void clearMmapHook() { _mmapHook = nullptr; }
  void clearMunmapHook() { _munmapHook = nullptr; }
  void clearCloseHook() { _closeHook = nullptr; }
  void clearIoctlHook() { _ioctlHook = nullptr; }
  void clearFopenHook() { _fopenHook = nullptr; }
  void clearFcloseHook() { _fcloseHook = nullptr; }
  void clearFreadHook() { _freadHook = nullptr; }
  void clearFseekHook() { _fseekHook = nullptr; }

  // Reset all hooks
  void resetAll()
  {
    clearOpenHook();
    clearOpenCreatHook();
    clearMmapHook();
    clearMunmapHook();
    clearCloseHook();
    clearIoctlHook();
    clearFopenHook();
    clearFcloseHook();
    clearFreadHook();
    clearFseekHook();
  }

private:
  MockSyscallsManager() = default;
  ~MockSyscallsManager() = default;
  MockSyscallsManager(const MockSyscallsManager &) = delete;
  MockSyscallsManager &operator=(const MockSyscallsManager &) = delete;

  // Hook function pointers
  OpenHook _openHook;
  OpenCreatHook _openCreatHook;
  MmapHook _mmapHook;
  MunmapHook _munmapHook;
  CloseHook _closeHook;
  IoctlHook _ioctlHook;
  FopenHook _fopenHook;
  FcloseHook _fcloseHook;
  FreadHook _freadHook;
  FseekHook _fseekHook;
};

} // namespace test
} // namespace ops
} // namespace trix
} // namespace backend
} // namespace onert

// Mock syscall implementations
int open(const char *pathname, int flags, ...);
void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
int munmap(void *addr, size_t length);
int close(int fd);
int ioctl(int fd, unsigned long request, ...);
FILE *fopen(const char *path, const char *mode);
int fclose(FILE *stream);
size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
int fseek(FILE *stream, long offset, int whence);

#endif // _MOCK_SYSCALLS_H_
