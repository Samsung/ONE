#ifndef HELPERS_HPP
#define HELPERS_HPP
#include <rtos.h>

#include <mutex>

#define LOCK_GUARD(x) std::lock_guard _lock_guard_instance##__LINE__(x)
#define with_lock(mtx) if (auto lock = std::lock_guard{mtx}; true)

#endif // HELPERS_HPP
