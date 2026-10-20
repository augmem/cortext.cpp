#include "aist_late_interaction.hpp"

#include "aist_embedded_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>

namespace cortext::internal
{

namespace
{

using RowMajorMatrix
    = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// ---------------------------------------------------------------- GGUF (F32)
template <typename T>
T
Read (std::istream &in)
{
  T value{};
  in.read (reinterpret_cast<char *> (&value), sizeof (T));
  if (!in)
    {
      throw std::runtime_error ("late-interaction GGUF truncated");
    }
  return value;
}

std::string
ReadString (std::istream &in)
{
  const auto n = Read<std::uint64_t> (in);
  if (n > (1U << 20))
    {
      throw std::runtime_error ("late-interaction GGUF string too long");
    }
  std::string s (static_cast<std::size_t> (n), '\0');
  in.read (s.data (), static_cast<std::streamsize> (n));
  if (!in)
    {
      throw std::runtime_error ("late-interaction GGUF truncated");
    }
  return s;
}

std::size_t
ScalarBytes (std::uint32_t type)
{
  switch (type)
    {
    case 0: case 1: case 7: return 1;
    case 2: case 3: return 2;
    case 4: case 5: case 6: return 4;
    case 10: case 11: case 12: return 8;
    default: return 0;
    }
}

void
SkipValue (std::istream &in, std::uint32_t type)
{
  if (type == 8)
    {
      (void)ReadString (in);
      return;
    }
  if (type == 9)
    {
      const auto elem = Read<std::uint32_t> (in);
      const auto count = Read<std::uint64_t> (in);
      for (std::uint64_t i = 0; i < count; ++i)
        {
          SkipValue (in, elem);
        }
      return;
    }
  const std::size_t bytes = ScalarBytes (type);
  if (bytes == 0)
    {
      throw std::runtime_error ("late-interaction GGUF: unknown value type");
    }
  in.seekg (static_cast<std::streamoff> (bytes), std::ios::cur);
}

struct GgufFile
{
  std::map<std::string, std::int64_t> ints;
  std::map<std::string, std::pair<std::vector<std::uint64_t>,
                                  std::vector<float>>> tensors;
};

GgufFile
ReadGguf (const std::filesystem::path &path)
{
  std::ifstream in (path, std::ios::binary);
  if (!in)
    {
      throw std::runtime_error ("late-interaction model not found: "
                                + path.string ());
    }
  char magic[4]{};
  in.read (magic, 4);
  if (!in || std::memcmp (magic, "GGUF", 4) != 0)
    {
      throw std::runtime_error ("late-interaction model is not GGUF: "
                                + path.string ());
    }
  const auto version = Read<std::uint32_t> (in);
  if (version < 2 || version > 3)
    {
      throw std::runtime_error ("late-interaction GGUF version unsupported");
    }
  const auto n_tensors = Read<std::uint64_t> (in);
  const auto n_kv = Read<std::uint64_t> (in);
  GgufFile file;
  std::uint64_t alignment = 32;
  for (std::uint64_t i = 0; i < n_kv; ++i)
    {
      const std::string key = ReadString (in);
      const auto type = Read<std::uint32_t> (in);
      if (type == 4)
        {
          const auto v = Read<std::uint32_t> (in);
          file.ints[key] = static_cast<std::int64_t> (v);
          if (key == "general.alignment")
            {
              alignment = v;
            }
        }
      else if (type == 5)
        {
          file.ints[key] = Read<std::int32_t> (in);
        }
      else
        {
          SkipValue (in, type);
        }
    }
  struct Info
  {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::uint32_t type = 0;
    std::uint64_t offset = 0;
  };
  std::vector<Info> infos;
  for (std::uint64_t i = 0; i < n_tensors; ++i)
    {
      Info info;
      info.name = ReadString (in);
      const auto n_dims = Read<std::uint32_t> (in);
      if (n_dims > 4)
        {
          throw std::runtime_error ("late-interaction GGUF: bad tensor rank");
        }
      for (std::uint32_t d = 0; d < n_dims; ++d)
        {
          info.shape.push_back (Read<std::uint64_t> (in));
        }
      info.type = Read<std::uint32_t> (in);
      info.offset = Read<std::uint64_t> (in);
      if (info.type != 0)
        {
          throw std::runtime_error (
              "late-interaction GGUF: only F32 tensors supported (" + info.name
              + ")");
        }
      infos.push_back (std::move (info));
    }
  const auto here = static_cast<std::uint64_t> (in.tellg ());
  const std::uint64_t data_start = (here + alignment - 1) / alignment * alignment;
  for (const Info &info : infos)
    {
      std::uint64_t n = 1;
      for (auto d : info.shape)
        {
          n *= d;
        }
      if (n > (1ULL << 26))
        {
          throw std::runtime_error ("late-interaction GGUF tensor too large");
        }
      std::vector<float> data (static_cast<std::size_t> (n));
      in.seekg (static_cast<std::streamoff> (data_start + info.offset));
      in.read (reinterpret_cast<char *> (data.data ()),
               static_cast<std::streamsize> (n * sizeof (float)));
      if (!in)
        {
          throw std::runtime_error ("late-interaction GGUF data truncated");
        }
      file.tensors[info.name] = { info.shape, std::move (data) };
    }
  return file;
}

/// GGUF shape is innermost-first: a [rows x cols] row-major matrix has
/// shape {cols, rows}.
Eigen::MatrixXf
Matrix (const GgufFile &file, const std::string &name, int rows, int cols)
{
  const auto it = file.tensors.find (name);
  if (it == file.tensors.end ())
    {
      throw std::runtime_error ("late-interaction GGUF missing tensor " + name);
    }
  const auto &[shape, data] = it->second;
  if (static_cast<int> (data.size ()) != rows * cols)
    {
      throw std::runtime_error ("late-interaction GGUF tensor shape " + name);
    }
  return Eigen::Map<const RowMajorMatrix> (data.data (), rows, cols);
}

float
Scalar (const GgufFile &file, const std::string &name)
{
  const auto it = file.tensors.find (name);
  if (it == file.tensors.end () || it->second.second.size () != 1)
    {
      throw std::runtime_error ("late-interaction GGUF missing scalar " + name);
    }
  return it->second.second[0];
}

int
IntMeta (const GgufFile &file, const std::string &key, int fallback)
{
  const auto it = file.ints.find (key);
  return it == file.ints.end () ? fallback : static_cast<int> (it->second);
}

/// int8 per-vector symmetric quantization of a unit vector (scale = max|x|/127).
void
QuantizeRow (const float *x, int dim, std::int8_t *out)
{
  float m = 0.0F;
  for (int i = 0; i < dim; ++i)
    {
      m = std::max (m, std::fabs (x[i]));
    }
  const float inv = m > 0.0F ? 127.0F / m : 0.0F;
  for (int i = 0; i < dim; ++i)
    {
      const float q = std::round (x[i] * inv);
      out[i] = static_cast<std::int8_t> (std::clamp (q, -127.0F, 127.0F));
    }
}

/// Spherical k-means of position rows (tools/reranker/codes.py): init at
/// floor(linspace(0, n-1, k)), up to `iters` iterations, argmax-cosine
/// assignment (first index on ties), centroid = mean of the raw rows (empty
/// clusters keep their previous centroid). An iteration that reassigns nothing
/// leaves the centroids unchanged, so stopping there gives the fixed-iteration
/// result. n <= k keeps the rows.
RowMajorMatrix
KMeansRows (const float *rows, int n, int dim, int k, int iters)
{
  Eigen::Map<const RowMajorMatrix> x (rows, n, dim);
  if (n <= k)
    {
      return x;
    }
  RowMajorMatrix xn = x;
  for (int i = 0; i < n; ++i)
    {
      const float norm = xn.row (i).norm ();
      if (norm > 0.0F)
        {
          xn.row (i) /= norm;
        }
    }
  RowMajorMatrix c (k, dim);
  const double step = static_cast<double> (n - 1) / static_cast<double> (k - 1);
  for (int j = 0; j < k; ++j)
    {
      int idx = static_cast<int> (std::floor (static_cast<double> (j) * step));
      if (j == k - 1)
        {
          idx = n - 1;
        }
      c.row (j) = x.row (std::clamp (idx, 0, n - 1));
    }
  std::vector<int> assign (static_cast<std::size_t> (n), -1);
  for (int it = 0; it < iters; ++it)
    {
      bool changed = false;
      RowMajorMatrix cn = c;
      for (int j = 0; j < k; ++j)
        {
          const float norm = cn.row (j).norm ();
          if (norm > 0.0F)
            {
              cn.row (j) /= norm;
            }
        }
      const RowMajorMatrix sim = xn * cn.transpose ();
      for (int i = 0; i < n; ++i)
        {
          int best = 0;
          for (int j = 1; j < k; ++j)
            {
              if (sim (i, j) > sim (i, best))
                {
                  best = j;
                }
            }
          changed |= assign[static_cast<std::size_t> (i)] != best;
          assign[static_cast<std::size_t> (i)] = best;
        }
      if (!changed)
        {
          break;
        }
      RowMajorMatrix sum = RowMajorMatrix::Zero (k, dim);
      std::vector<int> count (static_cast<std::size_t> (k), 0);
      for (int i = 0; i < n; ++i)
        {
          sum.row (assign[static_cast<std::size_t> (i)]) += x.row (i);
          ++count[static_cast<std::size_t> (assign[static_cast<std::size_t> (i)])];
        }
      for (int j = 0; j < k; ++j)
        {
          if (count[static_cast<std::size_t> (j)] > 0)
            {
              c.row (j) = sum.row (j)
                          / static_cast<float> (count[static_cast<std::size_t> (j)]);
            }
        }
    }
  return c;
}

} // namespace

std::unique_ptr<LateInteractionModel>
LateInteractionModel::Load (const std::filesystem::path &path)
{
  const GgufFile file = ReadGguf (path);
  auto model = std::unique_ptr<LateInteractionModel> (new LateInteractionModel ());
  model->dim_ = IntMeta (file, "li.dim", 0);
  if (model->dim_ <= 0 || model->dim_ > 1024)
    {
      throw std::runtime_error ("late-interaction GGUF: bad li.dim");
    }
  model->codes_per_member_ = IntMeta (file, "li.codes_per_member", 16);
  model->text_cap_ = IntMeta (file, "li.text_cap", 256);
  model->query_cap_ = IntMeta (file, "li.query_cap", 128);
  model->kmeans_iters_ = IntMeta (file, "li.kmeans_iters", 8);
  model->depth_ = IntMeta (file, "li.depth", 64);
  model->max_members_ = IntMeta (file, "li.max_members", 8);
  if (model->codes_per_member_ < 2 || model->codes_per_member_ > 255)
    {
      throw std::runtime_error ("late-interaction GGUF: bad codes_per_member");
    }
  const int d = model->dim_;
  const auto load_map = [&] (const std::string &name, int in)
      -> std::optional<TowerMap> {
    if (file.tensors.find (name) == file.tensors.end ())
      {
        return std::nullopt;
      }
    return TowerMap{ in, Matrix (file, name, d, in) };
  };
  model->text_ = load_map ("li.map.text", 384);
  model->image_ = load_map ("li.map.image", 960);
  model->audio_ = load_map ("li.map.audio", 960);
  model->global_ = Matrix (file, "li.glob", d, 256);
  model->gate_w_ = Matrix (file, "li.gate.weight", 1, d).row (0).transpose ();
  model->gate_b_ = Scalar (file, "li.gate.bias");
  model->global_bias_ = Scalar (file, "li.glob_bias");
  model->scale_ = Scalar (file, "li.scale");
  model->member_tau_ = std::exp (Scalar (file, "li.member_tau"));
  model->rank_w_ = Scalar (file, "li.rank_w");
  return model;
}

const LateInteractionModel::TowerMap *
LateInteractionModel::MapFor (const std::string &tower) const
{
  if (tower == "text" && text_)
    {
      return &*text_;
    }
  if (tower == "image" && image_)
    {
      return &*image_;
    }
  if (tower == "audio" && audio_)
    {
      return &*audio_;
    }
  return nullptr;
}

Eigen::MatrixXf
LateInteractionModel::MapRows (const TowerMap &map, const float *rows,
                               int count, int stride) const
{
  Eigen::Map<const RowMajorMatrix, 0, Eigen::OuterStride<>> x (
      rows, count, map.in, Eigen::OuterStride<> (stride));
  Eigen::MatrixXf y = x * map.weight.transpose ();  // [count x dim]
  for (int r = 0; r < count; ++r)
    {
      const float norm = y.row (r).norm ();
      if (norm > 0.0F)
        {
          y.row (r) /= norm;
        }
    }
  return y;
}

LateInteractionSet
LateInteractionModel::MemorySet (const EncoderPositions &positions) const
{
  LateInteractionSet out;
  const TowerMap *map = MapFor (positions.tower);
  if (map == nullptr || positions.count <= 0 || positions.dim != map->in)
    {
      return out;
    }
  const int n = positions.tower == "text"
                    ? std::min (positions.count, text_cap_)
                    : positions.count;
  const RowMajorMatrix codes = KMeansRows (positions.values.data (), n,
                                           positions.dim, codes_per_member_,
                                           kmeans_iters_);
  const Eigen::MatrixXf mapped = MapRows (*map, codes.data (),
                                          static_cast<int> (codes.rows ()),
                                          positions.dim);
  out.count = static_cast<int> (mapped.rows ());
  out.dim = dim_;
  out.codes.resize (static_cast<std::size_t> (out.count * dim_));
  std::vector<float> row (static_cast<std::size_t> (dim_));
  for (int r = 0; r < out.count; ++r)
    {
      for (int c = 0; c < dim_; ++c)
        {
          row[static_cast<std::size_t> (c)] = mapped (r, c);
        }
      QuantizeRow (row.data (), dim_,
                   out.codes.data () + static_cast<std::ptrdiff_t> (r * dim_));
    }
  return out;
}

LateInteractionSet
LateInteractionModel::QuerySet (const EncoderPositions &positions) const
{
  LateInteractionSet out;
  const TowerMap *map = MapFor (positions.tower);
  if (map == nullptr || positions.count <= 0 || positions.dim != map->in)
    {
      return out;
    }
  const int n = positions.tower == "text"
                    ? std::min (positions.count, query_cap_)
                    : positions.count;
  const Eigen::MatrixXf mapped = MapRows (*map, positions.values.data (), n,
                                          positions.dim);
  out.count = n;
  out.dim = dim_;
  out.values.resize (static_cast<std::size_t> (n * dim_));
  out.gate.resize (static_cast<std::size_t> (n));
  for (int r = 0; r < n; ++r)
    {
      for (int c = 0; c < dim_; ++c)
        {
          out.values[static_cast<std::size_t> (r * dim_ + c)] = mapped (r, c);
        }
      out.gate[static_cast<std::size_t> (r)]
          = mapped.row (r).dot (gate_w_.transpose ()) + gate_b_;
    }
  return out;
}

std::vector<unsigned char>
LateInteractionModel::Serialize (const LateInteractionSet &set)
{
  if (set.count <= 0 || set.dim <= 0 || set.count > 255 || set.dim > 65535
      || set.codes.size () != static_cast<std::size_t> (set.count * set.dim))
    {
      return {};
    }
  std::vector<unsigned char> out (4 + set.codes.size ());
  out[0] = 1;
  out[1] = static_cast<unsigned char> (set.count);
  out[2] = static_cast<unsigned char> (set.dim & 0xFF);
  out[3] = static_cast<unsigned char> ((set.dim >> 8) & 0xFF);
  std::memcpy (out.data () + 4, set.codes.data (), set.codes.size ());
  return out;
}

bool
LateInteractionModel::Deserialize (const unsigned char *data, std::size_t size,
                                   LateInteractionSet &out)
{
  out = {};
  if (data == nullptr || size < 4 || data[0] != 1)
    {
      return false;
    }
  const int count = data[1];
  const int dim = data[2] | (data[3] << 8);
  if (count <= 0 || dim <= 0
      || size != 4 + static_cast<std::size_t> (count) * static_cast<std::size_t> (dim))
    {
      return false;
    }
  out.count = count;
  out.dim = dim;
  out.codes.resize (static_cast<std::size_t> (count * dim));
  std::memcpy (out.codes.data (), data + 4, out.codes.size ());
  return true;
}

Eigen::VectorXf
LateInteractionModel::GlobalVector (const Eigen::VectorXf &global) const
{
  if (global.size () < global_.cols ())
    {
      return {};
    }
  Eigen::VectorXf g = global.head (global_.cols ());
  const float gn = g.norm ();
  if (gn <= 0.0F)
    {
      return {};
    }
  Eigen::VectorXf y = global_ * (g / gn);
  const float yn = y.norm ();
  if (yn > 0.0F)
    {
      y /= yn;
    }
  return y;
}

std::vector<float>
LateInteractionModel::Score (const LateInteractionSet &query,
                             const Eigen::VectorXf &query_global,
                             const std::vector<std::vector<Member>> &memories) const
{
  // Query vectors (float) + query global vector; gate softmax weights.
  const Eigen::VectorXf qg = GlobalVector (query_global);
  const int lq = (query.dim == dim_ ? query.count : 0);
  const int rows = lq + (qg.size () > 0 ? 1 : 0);
  std::vector<float> scores (memories.size (), 0.0F);
  if (rows == 0)
    {
      return scores;
    }
  Eigen::MatrixXf q (rows, dim_);
  Eigen::VectorXf logits (rows);
  for (int r = 0; r < lq; ++r)
    {
      for (int c = 0; c < dim_; ++c)
        {
          q (r, c) = query.values[static_cast<std::size_t> (r * dim_ + c)];
        }
      logits (r) = query.gate[static_cast<std::size_t> (r)];
    }
  if (qg.size () > 0)
    {
      q.row (lq) = qg.transpose ();
      logits (lq) = qg.dot (gate_w_) + gate_b_ + global_bias_;
    }
  const float lmax = logits.maxCoeff ();
  Eigen::VectorXf w = (logits.array () - lmax).exp ().matrix ();
  w /= w.sum ();

  // Every member vector of every memory (codes, then the stored global
  // vector, all as unit-norm int8 directions) goes into one matrix so the
  // similarities are a single product; members keep [begin, end) row ranges.
  struct Range
  {
    std::size_t memory = 0;
    const Member *member = nullptr;
    Eigen::Index begin = 0;
    Eigen::Index end = 0;
  };
  std::vector<Range> ranges;
  Eigen::Index total = 0;
  for (std::size_t m = 0; m < memories.size (); ++m)
    {
      const std::size_t take = std::min (
          memories[m].size (), static_cast<std::size_t> (max_members_));
      for (std::size_t s = 0; s < take; ++s)
        {
          const Member &member = memories[m][s];
          const Eigen::Index n
              = (member.codes.dim == dim_ ? member.codes.count : 0)
                + (member.global.size () >= global_.cols () ? 1 : 0);
          if (n > 0)
            {
              ranges.push_back ({ m, &member, total, total + n });
              total += n;
            }
        }
    }
  RowMajorMatrix mv (total, dim_);
  std::vector<std::int8_t> qrow (static_cast<std::size_t> (dim_));
  const auto put_unit = [&] (Eigen::Index row, const std::int8_t *codes) {
    double norm = 0.0;
    for (int c = 0; c < dim_; ++c)
      {
        norm += static_cast<double> (codes[c]) * codes[c];
      }
    const float inv = norm > 0.0 ? static_cast<float> (1.0 / std::sqrt (norm))
                                 : 0.0F;
    for (int c = 0; c < dim_; ++c)
      {
        mv (row, c) = static_cast<float> (codes[c]) * inv;
      }
  };
  for (const Range &range : ranges)
    {
      const Member &member = *range.member;
      Eigen::Index row = range.begin;
      if (member.codes.dim == dim_)
        {
          for (int r = 0; r < member.codes.count; ++r)
            {
              put_unit (row++, member.codes.codes.data ()
                                   + static_cast<std::ptrdiff_t> (r * dim_));
            }
        }
      if (member.global.size () >= global_.cols ())
        {
          // the stored global vector goes through the same int8 rounding as
          // codes (training applies it to every member vector)
          const Eigen::VectorXf mg = GlobalVector (member.global);
          if (mg.size () > 0)
            {
              QuantizeRow (mg.data (), dim_, qrow.data ());
              put_unit (row, qrow.data ());
            }
          else
            {
              mv.row (row).setZero ();
            }
        }
    }
  const RowMajorMatrix sim = q * mv.transpose ();  // [rows x total]

  std::vector<std::vector<float>> member_scores (memories.size ());
  for (const Range &range : ranges)
    {
      float acc = 0.0F;
      for (int r = 0; r < rows; ++r)
        {
          acc += w (r)
                 * sim.row (r).segment (range.begin, range.end - range.begin)
                       .maxCoeff ();
        }
      member_scores[range.memory].push_back (acc);
    }
  const float tau = member_tau_;
  for (std::size_t m = 0; m < memories.size (); ++m)
    {
      float mem = -1.0F;
      if (!member_scores[m].empty ())
        {
          const float mx = *std::max_element (member_scores[m].begin (),
                                              member_scores[m].end ());
          double sum = 0.0;
          for (float v : member_scores[m])
            {
              sum += std::exp ((v - mx) / tau);
            }
          mem = mx + tau * static_cast<float> (std::log (sum));
        }
      const float rank = static_cast<float> (std::log1p (static_cast<double> (m))
                                             / std::log (101.0));
      scores[m] = scale_ * mem + rank_w_ * rank;
    }
  return scores;
}

std::optional<std::filesystem::path>
ResolveLateInteractionModelPath (const std::filesystem::path &aist_model_path)
{
  if (const char *env = std::getenv ("CORTEXT_LI_MODEL_PATH"))
    {
      if (*env != '\0')
        {
          return std::filesystem::path (env);
        }
    }
  const std::filesystem::path candidate
      = aist_model_path.parent_path () / "AIST-87M-LI.gguf";
  if (std::filesystem::exists (candidate))
    {
      return candidate;
    }
  return MaterializeEmbeddedLateInteractionModel ();
}

} // namespace cortext::internal
