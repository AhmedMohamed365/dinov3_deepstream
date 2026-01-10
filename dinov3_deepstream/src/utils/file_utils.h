#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>

static inline void rstrip_inplace(std::string& s) {
  while (!s.empty()) {
    char c = s.back();
    if (c == '\n' || c == '\r' || c == ' ' || c == '\t') s.pop_back();
    else break;
  }
}

static inline std::vector<std::string> load_lines_txt(const std::string& path, bool skip_empty=true) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("Could not open file: " + path);

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(f, line)) {
    rstrip_inplace(line);
    if (skip_empty && line.empty()) continue;
    lines.push_back(line);
  }
  return lines;
}