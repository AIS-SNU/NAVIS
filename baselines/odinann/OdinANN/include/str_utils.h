#ifndef STR_UTILS_H_
#define STR_UTILS_H_

#include <ctype.h>

#include <algorithm>
#include <locale>
#include <string>
#include <vector>

// Attention: std::string can cause some construct-deconstruct cost
// Use slice & thread-local buffer in high-performance scenario.

inline std::string strip(const std::string &s) {
  std::string ret = s;
  ret.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
  ret.erase(std::find_if(ret.rbegin(), ret.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
            ret.end());
  return ret;
}

inline std::vector<std::string> split(const std::string &s, char delim) {
  // Not using stringstream, for it has poor multi-thread performance
  // Due to the creation of stringstream object using std::locale (globally locked)
  std::vector<std::string> elems;
  size_t l = 0, r = 0;
  while (r < s.size()) {
    if (s[r] == delim) {
      if (r - l > 0) {
        elems.push_back(s.substr(l, r - l));
      }
      l = r + 1;
    }
    r++;
  }
  if (r - l > 0) {
    elems.push_back(s.substr(l, r - l));
  }
  return elems;
}

#endif  // STR_UTILS_H_