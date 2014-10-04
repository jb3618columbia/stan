#ifndef TEST__UNIT__UTIL_HPP
#define TEST__UNIT__UTIL_HPP

#include <string>

#define EXPECT_THROW_MSG(expr, E, msg) \
  EXPECT_THROW(expr, T_exception);     \
  try {                                \
    (expr);                                     \
  } catch(const E& e) {                         \
    EXPECT_EQ(1, count_matches(msg, e.what()))  \
    << "expected message: " << msg << std::endl \
    << "found message:    " << e.what();        \
    return;                                     \
  }

int count_matches(const std::string& target,
                  const std::string& s) {
  if (target.size() == 0) return -1;  // error
  int count = 0;
  for (size_t pos = 0; (pos = s.find(target,pos)) != std::string::npos; pos += target.size())
    ++count;
  return count;
}

#endif
