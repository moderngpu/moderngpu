#pragma once
#include "types.hxx"
#include <cstdarg>
#include <string>

BEGIN_MGPU_NAMESPACE

namespace detail {

inline std::string stringprintf(const char* format, ...) {
  va_list args;
  va_start(args, format);
  int len = vsnprintf(0, 0, format, args);
  va_end(args);

  // allocate space.
  std::string text;
  text.resize(len);

  va_start(args, format);
  vsnprintf(&text[0], len + 1, format, args);
  va_end(args);

  return text;
}

} // namespace detail

END_MGPU_NAMESPACE

