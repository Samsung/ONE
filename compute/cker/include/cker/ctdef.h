#pragma once

#include <type_traits>

// Compile time definitions and related trait-like utilities

namespace nnfw
{
// This enum specifies all compile-time features that should be testable using the `is_defined` and
// `is_defined_v` templates below.
enum class Define
{
  PLATFORM_X86,
  PLATFORM_AARCH64,
  USE_NEON
};

// A compile-time feature-detection structure which defaults to false
// for all defines. This struct should be specialized for each Define enum value.
//
// This struct should be used the following way:
//
//   if constexpr (nnfw::is_defined<Define::PLATFORM_X86>::value) {}
template <Define> struct is_defined : std::false_type
{
};

// A helper template variable to be used the following way:
//
//   if constexpr (nnfw::is_defined_v<Define::PLATFORM_X86>) {}
template <Define def> inline constexpr const bool is_defined_v = is_defined<def>::value;

/* ********************************************************************************************** */

#if defined(CKER_X86_PLATFORM)
template <> struct is_defined<Define::PLATFORM_X86> : std::true_type
{
};
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
template <> struct is_defined<Define::PLATFORM_AARCH64> : std::true_type
{
};
#endif

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
template <> struct is_defined<Define::USE_NEON> : std::true_type
{
};
#endif

} // namespace nnfw
