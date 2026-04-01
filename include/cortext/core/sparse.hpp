#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace cortext::core
{

inline int
SparseKeySize (double F)
{
  const double t = (F < 0.0) ? 0.0 : (F > 1.0 ? 1.0 : F);
  const int k = static_cast<int> (std::round (16.0 + (64.0 - 16.0) * t));
  return std::max (8, k);
}

inline std::string
SparseKey (const Eigen::VectorXf &v, int k)
{
  if (v.size () == 0 || k <= 0)
    {
      return {};
    }

  std::vector<std::pair<float, int>> mags;
  mags.reserve (static_cast<size_t> (v.size ()));
  for (int i = 0; i < v.size (); ++i)
    {
      mags.emplace_back (std::fabs (v[i]), i);
    }

  if (k > static_cast<int> (mags.size ()))
    k = static_cast<int> (mags.size ());

  std::partial_sort (
      mags.begin (), mags.begin () + k, mags.end (),
      [] (const auto &a, const auto &b) { return a.first > b.first; });

  mags.resize (static_cast<size_t> (k));
  std::vector<int> indices;
  indices.reserve (mags.size ());
  for (const auto &p : mags)
    {
      indices.push_back (p.second);
    }
  std::sort (indices.begin (), indices.end ());

  std::ostringstream oss;
  for (size_t i = 0; i < indices.size (); ++i)
    {
      const int idx = indices[i];
      const bool positive = v[idx] >= 0.0f;
      if (i > 0)
        oss << ",";
      oss << idx << (positive ? "p" : "n");
    }
  return oss.str ();
}

} // namespace cortext::core
