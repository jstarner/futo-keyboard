#ifndef LATINIME_BARK_FORMAT_H
#define LATINIME_BARK_FORMAT_H

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "bark_ops.h"

// Reader for .bark files ("bark-levels-v2" and "bark-levels-v3"): an 8-byte
// little-endian header length, a JSON header, then one blob per tensor.
//   levels2: per-row fp16 levels plus one ANS-coded code per weight, with 16
//            column classes that each have their own symbol model
//   ansintN: per-row symmetric N-bit integers, ANS coded, with fp16 row scales
//   raw:     fp16 (or int64/int32) values
// Side blobs (levels, scales, raw values) are xz-compressed when flagged in v3.
// Decoding is exact; it mirrors bark_loader.py from the model repository and
// the constriction 0.5 AnsCoder / Categorical(perfect=False) it uses.

namespace bark {

struct Json {
    enum class Kind { Null, Bool, Number, String, Array, Object };

    Kind kind = Kind::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Json> array;
    std::vector<std::pair<std::string, Json>> object;

    // Returns a Null value if missing
    const Json &operator[](const std::string &key) const;

    bool is_null() const { return kind == Kind::Null; }
    int as_int() const { return (int)number; }

    static bool parse(const char *data, size_t size, Json &out, std::string &error);
};

struct Tensor {
    std::vector<int> shape;
    Matrix matrix;
};

using TensorMap = std::unordered_map<std::string, Tensor>;

bool has_bark_header(const uint8_t *data, size_t size);

// Decodes every float tensor of the file. config receives the model config.
bool read_bark_file(const uint8_t *data, size_t size, ThreadPool &pool, TensorMap &tensors, Json &config,
                    std::string &error);

} // namespace bark

#endif //LATINIME_BARK_FORMAT_H
