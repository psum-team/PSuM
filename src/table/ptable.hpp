#ifndef PSUM_TABLE_PTABLE_HPP
#define PSUM_TABLE_PTABLE_HPP

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace psum::table {

template <class... Ts>
class VariadicTable {
  static constexpr size_t N = sizeof...(Ts);
  std::vector<std::string> headers_;
  std::vector<std::vector<std::string>> rows_;
  std::vector<bool> right_{std::is_arithmetic_v<Ts>...};

  template <class T> static std::string cell(const T& v) { std::ostringstream o; o << v; return o.str(); }
  static std::string fit(const std::string& v, size_t w, bool right) {
    std::string pad(w >= v.size() ? w - v.size() : 0, ' ');
    return right ? pad + v : v + pad;
  }

public:
  explicit VariadicTable(std::vector<std::string> headers, unsigned int = 1)
      : headers_(std::move(headers)) {}

  void addRow(Ts const&... args) { rows_.push_back({cell(args)...}); }

  template <class Stream> void print(Stream& s) {
    std::vector<size_t> w(N);
    for (size_t c = 0; c < N; ++c) {
      w[c] = headers_[c].size();
      for (auto& r : rows_) w[c] = std::max(w[c], r[c].size());
    }
    size_t total = N + 1;
    for (size_t c = 0; c < N; ++c) total += w[c] + 2;
    const std::string line(total, '-');

    s << line << "\n|";
    for (size_t c = 0; c < N; ++c) {
      long half = long(w[c] / 2) - long(headers_[c].size() / 2);
      if (half < 0) half = 0;
      s << " " << fit(std::string(size_t(half), ' ') + headers_[c], w[c], false) << " |";
    }
    s << "\n" << line << "\n";
    for (auto& r : rows_) {
      s << "|";
      for (size_t c = 0; c < N; ++c) s << " " << fit(r[c], w[c], right_[c]) << " |";
      s << "\n";
    }
    s << line << "\n";
  }
};

} // namespace psum::table

#endif