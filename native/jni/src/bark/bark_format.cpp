#include "bark_format.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "third_party/xz-embedded/xz.h"

namespace bark {

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

const Json &Json::operator[](const std::string &key) const {
    static const Json null_value;
    for(const auto &kv : object) {
        if(kv.first == key) return kv.second;
    }
    return null_value;
}

namespace {

class JsonParser {
public:
    JsonParser(const char *data, size_t size) : p(data), end(data + size) {}

    bool parse(Json &out, std::string &error) {
        if(!value(out, 0)) {
            error = "Invalid JSON in model header";
            return false;
        }
        return true;
    }

private:
    const char *p;
    const char *end;

    void skip_ws() {
        while(p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    }

    bool literal(const char *word) {
        const size_t n = strlen(word);
        if((size_t)(end - p) < n || memcmp(p, word, n) != 0) return false;
        p += n;
        return true;
    }

    static void append_utf8(std::string &s, uint32_t cp) {
        if(cp < 0x80) {
            s += (char)cp;
        } else if(cp < 0x800) {
            s += (char)(0xC0 | (cp >> 6));
            s += (char)(0x80 | (cp & 0x3F));
        } else if(cp < 0x10000) {
            s += (char)(0xE0 | (cp >> 12));
            s += (char)(0x80 | ((cp >> 6) & 0x3F));
            s += (char)(0x80 | (cp & 0x3F));
        } else {
            s += (char)(0xF0 | (cp >> 18));
            s += (char)(0x80 | ((cp >> 12) & 0x3F));
            s += (char)(0x80 | ((cp >> 6) & 0x3F));
            s += (char)(0x80 | (cp & 0x3F));
        }
    }

    bool hex4(uint32_t &out) {
        if(end - p < 4) return false;
        out = 0;
        for(int i = 0; i < 4; i++) {
            const char c = *p++;
            out <<= 4;
            if(c >= '0' && c <= '9') out |= c - '0';
            else if(c >= 'a' && c <= 'f') out |= c - 'a' + 10;
            else if(c >= 'A' && c <= 'F') out |= c - 'A' + 10;
            else return false;
        }
        return true;
    }

    bool string(std::string &out) {
        if(p >= end || *p != '"') return false;
        p++;
        while(p < end && *p != '"') {
            if(*p != '\\') {
                out += *p++;
                continue;
            }
            p++;
            if(p >= end) return false;
            const char c = *p++;
            switch(c) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp;
                    if(!hex4(cp)) return false;
                    if(cp >= 0xD800 && cp < 0xDC00 && end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                        p += 2;
                        uint32_t lo;
                        if(!hex4(lo)) return false;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: return false;
            }
        }
        if(p >= end) return false;
        p++;
        return true;
    }

    bool value(Json &out, int depth) {
        if(depth > 64) return false;
        skip_ws();
        if(p >= end) return false;

        switch(*p) {
            case '{': {
                out.kind = Json::Kind::Object;
                p++;
                skip_ws();
                if(p < end && *p == '}') { p++; return true; }
                while(true) {
                    skip_ws();
                    std::string key;
                    if(!string(key)) return false;
                    skip_ws();
                    if(p >= end || *p != ':') return false;
                    p++;
                    Json v;
                    if(!value(v, depth + 1)) return false;
                    out.object.emplace_back(std::move(key), std::move(v));
                    skip_ws();
                    if(p < end && *p == ',') { p++; continue; }
                    if(p < end && *p == '}') { p++; return true; }
                    return false;
                }
            }
            case '[': {
                out.kind = Json::Kind::Array;
                p++;
                skip_ws();
                if(p < end && *p == ']') { p++; return true; }
                while(true) {
                    Json v;
                    if(!value(v, depth + 1)) return false;
                    out.array.push_back(std::move(v));
                    skip_ws();
                    if(p < end && *p == ',') { p++; continue; }
                    if(p < end && *p == ']') { p++; return true; }
                    return false;
                }
            }
            case '"':
                out.kind = Json::Kind::String;
                return string(out.string);
            case 't':
                out.kind = Json::Kind::Bool;
                out.boolean = true;
                return literal("true");
            case 'f':
                out.kind = Json::Kind::Bool;
                out.boolean = false;
                return literal("false");
            case 'n':
                out.kind = Json::Kind::Null;
                return literal("null");
            default: {
                const char *start = p;
                while(p < end && (strchr("+-.eE", *p) != nullptr || (*p >= '0' && *p <= '9'))) p++;
                if(p == start) return false;
                const std::string num(start, p);
                char *num_end = nullptr;
                out.kind = Json::Kind::Number;
                out.number = strtod(num.c_str(), &num_end);
                return num_end == num.c_str() + num.size();
            }
        }
    }
};

} // namespace

bool Json::parse(const char *data, size_t size, Json &out, std::string &error) {
    return JsonParser(data, size).parse(out, error);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

constexpr int NUM_CLASSES = 16;
constexpr uint64_t MAX_HEADER_SIZE = 64 * 1024 * 1024;

float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if(exp == 0) {
        if(mant == 0) {
            bits = sign;
        } else {
            // Subnormal: normalize
            exp = 127 - 15 + 1;
            while((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if(exp == 0x1F) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

std::vector<float> halves_to_floats(const uint8_t *data, size_t count) {
    std::vector<float> out(count);
    for(size_t i = 0; i < count; i++) {
        uint16_t h;
        memcpy(&h, data + i * 2, 2);
        out[i] = half_to_float(h);
    }
    return out;
}

// expected = 0 accepts any size
bool xz_decompress(const uint8_t *in, size_t in_size, size_t expected, std::vector<uint8_t> &out) {
    struct xz_dec *dec = xz_dec_init(XZ_DYNALLOC, 1u << 26);
    if(dec == nullptr) return false;

    out.resize(expected ? expected + 1 : in_size * 4 + 4096);
    struct xz_buf buf = {};
    buf.in = in;
    buf.in_size = in_size;
    buf.out = out.data();
    buf.out_size = out.size();

    enum xz_ret ret;
    while(true) {
        ret = xz_dec_run(dec, &buf);
        if(ret != XZ_OK || buf.out_pos < buf.out_size) break;
        out.resize(out.size() * 2);
        buf.out = out.data();
        buf.out_size = out.size();
    }
    xz_dec_end(dec);

    if(ret != XZ_STREAM_END || (expected && buf.out_pos != expected)) return false;
    out.resize(buf.out_pos);
    return true;
}

// A blob that is either stored as is or xz-compressed
bool side_blob(const uint8_t *data, size_t size, bool compressed, size_t expected, std::vector<uint8_t> &out) {
    if(compressed) return xz_decompress(data, size, expected, out);
    if(size != expected) return false;
    out.assign(data, data + size);
    return true;
}

// constriction's DefaultContiguousCategoricalEntropyModel (24-bit precision)
// built with from_floating_point_probabilities_fast, from integer counts the
// way bark_loader.py passes them (counts / counts.sum() as float64)
constexpr int ANS_PRECISION = 24;

bool build_cdf(const uint32_t *counts, int n, std::vector<uint32_t> &cdf) {
    if(n < 2) return false;

    double total = 0.0;
    for(int i = 0; i < n; i++) total += (double)counts[i]; // integers, so exact in any order
    if(total <= 0.0) return false;

    std::vector<double> probs(n);
    for(int i = 0; i < n; i++) probs[i] = (double)counts[i] / total;

    double normalization = 0.0;
    for(int i = 0; i < n; i++) normalization += probs[i];

    const uint32_t free_weight = (1u << ANS_PRECISION) - (uint32_t)n;
    const double scale = (double)free_weight / normalization;

    cdf.resize(n + 1);
    double cumulative = 0.0;
    for(int i = 0; i < n; i++) {
        const double scaled = cumulative * scale;
        cdf[i] = (uint32_t)scaled + (uint32_t)i;
        cumulative += probs[i];
    }
    cdf[n] = 1u << ANS_PRECISION;
    return true;
}

// constriction's stack::AnsCoder<u32, u64> decoding from a word vector (words
// are popped from the end)
class AnsDecoder {
public:
    bool init(const uint8_t *data, size_t size) {
        if(size % 4 != 0) return false;
        words.resize(size / 4);
        if(size) memcpy(words.data(), data, size);
        n = words.size();

        state = 0;
        if(n == 0) return true;
        const uint32_t first = words[--n];
        if(first == 0) return false;
        state = first;
        while(n > 0) {
            state = (state << 32) | words[--n];
            if(state >= (1ull << 32)) break;
        }
        return true;
    }

    inline int decode(const uint32_t *cdf, int n_symbols) {
        const uint32_t quantile = (uint32_t)(state & ((1u << ANS_PRECISION) - 1));

        int symbol;
        if(n_symbols <= 8) {
            symbol = 0;
            for(int i = 1; i < n_symbols; i++) symbol += cdf[i] <= quantile;
        } else {
            // Largest symbol with cdf[symbol] <= quantile
            int lo = 0, hi = n_symbols;
            while(hi - lo > 1) {
                const int mid = (lo + hi) / 2;
                if(cdf[mid] <= quantile) lo = mid;
                else hi = mid;
            }
            symbol = lo;
        }

        const uint32_t left = cdf[symbol];
        const uint32_t probability = cdf[symbol + 1] - left;
        state = (state >> ANS_PRECISION) * probability + (quantile - left);
        if(state < (1ull << 32) && n > 0) {
            state = (state << 32) | words[--n];
        }
        return symbol;
    }

private:
    std::vector<uint32_t> words;
    size_t n = 0;
    uint64_t state = 0;
};

struct Entry {
    std::string name;
    std::string kind;
    std::vector<int> shape;
    size_t offset = 0;

    // raw
    std::string dtype;
    size_t b = 0;
    bool zr = false;

    // levels2
    int K = 0;
    size_t lev = 0;
    bool zl = false;
    size_t meta = 0;

    // ansint
    int bits = 0;
    size_t s = 0;
    bool zs = false;

    size_t numel() const {
        size_t n = 1;
        for(int d : shape) n *= (size_t)d;
        return n;
    }
};

bool decode_raw(const Entry &e, const uint8_t *data, Tensor &out, std::string &error) {
    const size_t numel = e.numel();
    std::vector<uint8_t> bytes;
    if(!side_blob(data + e.offset, e.b, e.zr, numel * 2, bytes)) {
        error = "Corrupt raw tensor " + e.name;
        return false;
    }

    out.shape = e.shape;
    Matrix &m = out.matrix;
    m.type = Matrix::Type::F32;
    m.rows = e.shape.empty() ? 1 : e.shape[0];
    m.cols = m.rows == 0 ? 0 : (int)(numel / m.rows);
    m.f32 = halves_to_floats(bytes.data(), numel);
    return true;
}

bool decode_ansint(const Entry &e, const uint8_t *data, Tensor &out, std::string &error) {
    const uint8_t *p = data + e.offset;
    const size_t numel = e.numel();

    std::vector<uint8_t> scale_bytes;
    if(!side_blob(p, e.s, e.zs, e.zs ? 0 : e.s, scale_bytes)) {
        error = "Corrupt scales of " + e.name;
        return false;
    }
    p += e.s;

    const size_t rows = scale_bytes.size() / 2;
    if(rows == 0 || numel % rows != 0 || e.meta % 4 != 0 || e.bits < 2 || e.bits > 8) {
        error = "Bad integer tensor " + e.name;
        return false;
    }

    const int n_symbols = (int)(e.meta / 4);
    std::vector<uint32_t> counts(n_symbols);
    memcpy(counts.data(), p, e.meta);
    p += e.meta;

    std::vector<uint32_t> cdf;
    AnsDecoder ans;
    if(!build_cdf(counts.data(), n_symbols, cdf) || !ans.init(p, e.b)) {
        error = "Bad entropy coder data in " + e.name;
        return false;
    }

    const int offset = (1 << (e.bits - 1)) - 1;
    out.shape = e.shape;
    Matrix &m = out.matrix;
    m.type = Matrix::Type::I8;
    m.rows = (int)rows;
    m.cols = (int)(numel / rows);
    m.scale = halves_to_floats(scale_bytes.data(), rows);
    m.i8.resize(numel);
    for(size_t i = 0; i < numel; i++) {
        const int q = ans.decode(cdf.data(), n_symbols) - offset;
        if(q < -128 || q > 127) {
            error = "Integer out of range in " + e.name;
            return false;
        }
        m.i8[i] = (int8_t)q;
    }
    return true;
}

bool decode_levels(const Entry &e, const uint8_t *data, Tensor &out, std::string &error) {
    const uint8_t *p = data + e.offset;
    const int K = e.K;
    if(e.shape.empty() || K < 1 || K > 7) {
        error = "Bad levels tensor " + e.name;
        return false;
    }

    const int rows = e.shape[0];
    const int cols = (int)(e.numel() / rows);
    const int n_symbols = 2 * K + 1;

    std::vector<uint8_t> lev_bytes;
    if(!side_blob(p, e.lev, e.zl, (size_t)rows * K * 2, lev_bytes)) {
        error = "Corrupt levels of " + e.name;
        return false;
    }
    p += e.lev;

    const size_t class_bytes = ((size_t)cols * 4 + 7) / 8;
    if(e.meta != class_bytes + (size_t)NUM_CLASSES * n_symbols * 4) {
        error = "Bad metadata in " + e.name;
        return false;
    }
    const uint8_t *meta = p;
    p += e.meta;

    // 4-bit class per column, most significant nibble first
    std::vector<std::vector<int>> class_columns(NUM_CLASSES);
    for(int c = 0; c < cols; c++) {
        const uint8_t byte = meta[c / 2];
        const int cls = (c % 2 == 0) ? (byte >> 4) : (byte & 0xF);
        class_columns[cls].push_back(c);
    }

    std::vector<uint32_t> counts((size_t)NUM_CLASSES * n_symbols);
    memcpy(counts.data(), meta + class_bytes, counts.size() * 4);

    out.shape = e.shape;
    Matrix &m = out.matrix;
    m.type = Matrix::Type::Levels;
    m.rows = rows;
    m.cols = cols;

    // Pack as many base-(2K+1) codes into a byte as fit
    int per_byte = 1;
    for(int v = n_symbols * n_symbols; v <= 256; v *= n_symbols) per_byte++;
    m.codes_per_byte = per_byte;
    m.row_bytes = (cols + per_byte - 1) / per_byte;
    m.packed.assign((size_t)rows * m.row_bytes, 0);

    m.unpack.resize(256 * per_byte);
    for(int b = 0; b < 256; b++) {
        int v = b;
        for(int j = 0; j < per_byte; j++) {
            // Bytes that cannot occur decode to code 0 (a zero weight)
            m.unpack[b * per_byte + j] = (uint8_t)(v % n_symbols);
            v /= n_symbols;
        }
    }

    int place[8] = {1};
    for(int j = 1; j < per_byte; j++) place[j] = place[j - 1] * n_symbols;

    // Row lookup tables: code 0 -> 0, 1..K -> +level, K+1..2K -> -level
    const std::vector<float> levels = halves_to_floats(lev_bytes.data(), (size_t)rows * K);
    m.lut_stride = n_symbols;
    m.lut.resize((size_t)rows * n_symbols);
    for(int r = 0; r < rows; r++) {
        float *l = &m.lut[(size_t)r * n_symbols];
        l[0] = 0.0f;
        for(int k = 0; k < K; k++) {
            const float v = levels[(size_t)r * K + k];
            l[1 + k] = v;
            l[1 + K + k] = -v;
        }
    }

    AnsDecoder ans;
    if(!ans.init(p, e.b)) {
        error = "Bad entropy coder data in " + e.name;
        return false;
    }

    // Classes in order; within a class, column by column, every row
    std::vector<uint32_t> cdf;
    for(int cls = 0; cls < NUM_CLASSES; cls++) {
        if(class_columns[cls].empty()) continue;
        if(!build_cdf(&counts[(size_t)cls * n_symbols], n_symbols, cdf)) {
            error = "Bad symbol model in " + e.name;
            return false;
        }

        for(int c : class_columns[cls]) {
            const int byte = c / per_byte;
            const uint8_t mult = (uint8_t)place[c % per_byte];
            uint8_t *dst = &m.packed[byte];
            for(int r = 0; r < rows; r++) {
                const int code = ans.decode(cdf.data(), n_symbols);
                dst[(size_t)r * m.row_bytes] += (uint8_t)(code * mult);
            }
        }
    }
    return true;
}

bool size_field(const Json &j, size_t &out) {
    if(j.kind != Json::Kind::Number || j.number < 0) return false;
    out = (size_t)j.number;
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// File
// ---------------------------------------------------------------------------

bool has_bark_header(const uint8_t *data, size_t size) {
    if(size < 64) return false;
    uint64_t n;
    memcpy(&n, data, 8);
    if(n < 16 || n > MAX_HEADER_SIZE || n + 8 > size) return false;

    // The header starts with the format name
    const char *json = reinterpret_cast<const char *>(data + 8);
    const size_t probe = std::min<size_t>(n, 64);
    const std::string start(json, probe);
    return start[0] == '{' && start.find("\"bark-levels-v") != std::string::npos;
}

bool read_bark_file(const uint8_t *data, size_t size, ThreadPool &pool, TensorMap &tensors, Json &config,
                    std::string &error) {
    if(!has_bark_header(data, size)) {
        error = "Not a Bark model file";
        return false;
    }

    uint64_t header_size;
    memcpy(&header_size, data, 8);

    Json head;
    if(!Json::parse(reinterpret_cast<const char *>(data + 8), header_size, head, error)) return false;

    const std::string &format = head["format"].string;
    if(format != "bark-levels-v2" && format != "bark-levels-v3") {
        error = "Unsupported Bark format " + format;
        return false;
    }

    const std::string &config_str = head["config"].string;
    if(!Json::parse(config_str.data(), config_str.size(), config, error)) return false;

    std::vector<Entry> entries;
    size_t offset = 8 + header_size;
    for(const Json &j : head["index"].array) {
        Entry e;
        e.name = j["n"].string;
        e.kind = j["k"].string;
        for(const Json &d : j["shape"].array) e.shape.push_back(d.as_int());
        e.offset = offset;

        bool ok = true;
        size_t total = 0;
        if(e.kind == "raw") {
            e.dtype = j["dtype"].string;
            e.zr = j["zr"].boolean;
            ok = size_field(j["b"], e.b);
            total = e.b;
        } else if(e.kind == "levels2") {
            e.K = j["K"].as_int();
            e.zl = j["zl"].boolean;
            ok = size_field(j["lev"], e.lev) && size_field(j["meta"], e.meta) && size_field(j["b"], e.b);
            total = e.lev + e.meta + e.b;
        } else if(e.kind.rfind("ansint", 0) == 0) {
            e.bits = atoi(e.kind.c_str() + 6);
            e.zs = j["zs"].boolean;
            ok = size_field(j["s"], e.s) && size_field(j["meta"], e.meta) && size_field(j["b"], e.b);
            total = e.s + e.meta + e.b;
        } else {
            error = "Unknown tensor encoding " + e.kind;
            return false;
        }

        if(!ok || total > size - offset) {
            error = "Truncated or invalid model file at " + e.name;
            return false;
        }
        offset += total;

        // Integer tensors (BatchNorm step counters) are not needed
        if(e.kind == "raw" && (e.dtype == "torch.int64" || e.dtype == "torch.int32")) continue;
        entries.push_back(std::move(e));
    }

    xz_crc32_init();
    xz_crc64_init();

    // Largest first, so the long decodes don't end up last on one thread
    std::vector<size_t> order(entries.size());
    for(size_t i = 0; i < order.size(); i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return entries[a].numel() > entries[b].numel();
    });

    std::vector<Tensor> decoded(entries.size());
    std::atomic<bool> failed{false};
    std::mutex error_mutex;

    pool.parallel_for((int)order.size(), [&](int i) {
        if(failed) return;
        const Entry &e = entries[order[i]];
        std::string err;
        bool ok;
        if(e.kind == "raw") ok = decode_raw(e, data, decoded[order[i]], err);
        else if(e.kind == "levels2") ok = decode_levels(e, data, decoded[order[i]], err);
        else ok = decode_ansint(e, data, decoded[order[i]], err);

        if(!ok) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if(!failed) error = err;
            failed = true;
        }
    });

    if(failed) return false;

    for(size_t i = 0; i < entries.size(); i++) {
        tensors[entries[i].name] = std::move(decoded[i]);
    }
    return true;
}

} // namespace bark
