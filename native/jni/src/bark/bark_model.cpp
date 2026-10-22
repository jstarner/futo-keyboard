#include "bark.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <memory>

#include "bark_format.h"
#include "bark_ops.h"

// Inference for Parakeet TDT models as implemented by transformers'
// ParakeetForTDT and ParakeetFeatureExtractor, which bark_loader.py evaluates with.

namespace bark {

namespace {

constexpr int SAMPLE_RATE = 16000;
constexpr int N_FFT = 512;
constexpr int WIN_LENGTH = 400;
constexpr int HOP_LENGTH = 160;
constexpr float PREEMPHASIS = 0.97f;
constexpr float LOG_ZERO_GUARD = 5.9604644775390625e-08f; // 2^-24
constexpr float NORM_EPSILON = 1e-5f;
constexpr float LAYER_NORM_EPS = 1e-5f;
constexpr float BATCH_NORM_EPS = 1e-5f;

// bark_loader.transcribe adds 0.25 s of silence before the audio (the evaluated setting)
constexpr int LEADING_SILENCE = SAMPLE_RATE / 4;

constexpr size_t MAX_SAMPLES = 60 * SAMPLE_RATE;

// Generation stops at this token (generation_config.json of parakeet-tdt-0.6b-v3)
constexpr int EOS_TOKEN = 3;

struct Config {
    int hidden = 0;
    int intermediate = 0;
    int layers = 0;
    int heads = 0;
    int conv_kernel = 0;
    int sub_channels = 0;
    int mel_bins = 0;
    int dec_hidden = 0;
    int dec_layers = 0;
    int vocab_size = 0; // including blank
    int blank = 0;
    int max_symbols = 0;
    std::vector<int> durations;
};

struct Layer {
    const Matrix *ff1_in = nullptr, *ff1_out = nullptr, *ff2_in = nullptr, *ff2_out = nullptr;
    const Matrix *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *rel_k = nullptr;
    const Matrix *pw1 = nullptr, *pw2 = nullptr;
    std::vector<float> norm_ff1_w, norm_ff1_b, norm_att_w, norm_att_b, norm_conv_w, norm_conv_b,
            norm_ff2_w, norm_ff2_b, norm_out_w, norm_out_b;
    std::vector<float> bias_u, bias_v;
    // Depthwise convolution with BatchNorm folded in
    std::vector<float> dw_weight, dw_bias;
};

int floor_div(int a, int b) {
    return (a >= 0) ? a / b : -((-a + b - 1) / b);
}

// Length after a kernel 3, stride 2, padding 1 convolution
int conv_out_len(int n) {
    return floor_div(n - 1, 2) + 1;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, norm="slaney"), Slaney mel scale
std::vector<float> mel_filterbank(int n_mels) {
    const int n_bins = N_FFT / 2 + 1;
    const double f_sp = 200.0 / 3.0;
    const double min_log_hz = 1000.0;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep = std::log(6.4) / 27.0;

    auto hz_to_mel = [&](double f) {
        return f >= min_log_hz ? min_log_mel + std::log(f / min_log_hz) / logstep : f / f_sp;
    };
    auto mel_to_hz = [&](double m) {
        return m >= min_log_mel ? min_log_hz * std::exp(logstep * (m - min_log_mel)) : f_sp * m;
    };

    const double mel_min = hz_to_mel(0.0);
    const double mel_max = hz_to_mel(SAMPLE_RATE / 2.0);
    std::vector<double> mel_f(n_mels + 2);
    for(int i = 0; i < n_mels + 2; i++) {
        mel_f[i] = mel_to_hz(mel_min + (mel_max - mel_min) * i / (n_mels + 1));
    }

    std::vector<float> weights((size_t)n_mels * n_bins);
    for(int m = 0; m < n_mels; m++) {
        const double lower_width = mel_f[m + 1] - mel_f[m];
        const double upper_width = mel_f[m + 2] - mel_f[m + 1];
        const double enorm = 2.0 / (mel_f[m + 2] - mel_f[m]);
        for(int k = 0; k < n_bins; k++) {
            const double freq = (double)SAMPLE_RATE / 2.0 * k / (n_bins - 1);
            const double lower = (freq - mel_f[m]) / lower_width;
            const double upper = (mel_f[m + 2] - freq) / upper_width;
            weights[(size_t)m * n_bins + k] = (float)(std::max(0.0, std::min(lower, upper)) * enorm);
        }
    }
    return weights;
}

// In-place radix-2 FFT
void fft(std::vector<std::complex<double>> &a, const std::vector<std::complex<double>> &twiddles) {
    const size_t n = a.size();
    for(size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for(; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if(i < j) std::swap(a[i], a[j]);
    }
    for(size_t len = 2; len <= n; len <<= 1) {
        const size_t step = n / len;
        for(size_t i = 0; i < n; i += len) {
            for(size_t j = 0; j < len / 2; j++) {
                const std::complex<double> u = a[i + j];
                const std::complex<double> v = a[i + j + len / 2] * twiddles[j * step];
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
            }
        }
    }
}

} // namespace

struct Model {
    Config config;
    TensorMap tensors;
    std::unique_ptr<ThreadPool> pool;

    std::vector<float> mel_filters;
    std::vector<float> window;
    std::vector<std::complex<double>> twiddles;

    // Subsampling
    std::vector<float> sub0_w, sub0_b, sub2_w, sub2_b, sub5_w, sub5_b;
    std::vector<float> sub3_b, sub6_b, sub_linear_b;
    const Matrix *sub3_w = nullptr, *sub6_w = nullptr, *sub_linear_w = nullptr;

    std::vector<Layer> layers;

    const Matrix *enc_proj_w = nullptr;
    std::vector<float> enc_proj_b;

    const Matrix *embedding = nullptr;
    std::vector<const Matrix *> lstm_w_ih, lstm_w_hh;
    std::vector<std::vector<float>> lstm_b;
    const Matrix *dec_proj_w = nullptr;
    std::vector<float> dec_proj_b;
    const Matrix *joint_w = nullptr;
    std::vector<float> joint_b;

    bool cancelled(const volatile int *cancel) const { return cancel != nullptr && *cancel != 0; }

    std::vector<float> log_mel(const std::vector<float> &audio, int &n_frames, int &n_valid) const;
    bool encode(const std::vector<float> &mel, int n_frames, int n_valid, const volatile int *cancel,
                std::vector<float> &out, int &T);
    void subsample(const std::vector<float> &mel, int n_frames, int n_valid, std::vector<float> &out, int &T);
    void encoder_layer(const Layer &L, std::vector<float> &x, int T, const std::vector<float> &pos);
    void decode(const std::vector<float> &enc, int T, int T_max, const volatile int *cancel,
                std::vector<int> &tokens);
};

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

namespace {

class Loader {
public:
    Loader(TensorMap &tensors, std::string &error) : tensors(tensors), error(error) {}

    const Matrix *matrix(const std::string &name, int rows, int cols) {
        auto it = tensors.find(name);
        if(it == tensors.end()) return fail("Missing tensor " + name);
        const Matrix &m = it->second.matrix;
        if(m.rows != rows || m.cols != cols) return fail("Unexpected shape of " + name);
        return &m;
    }

    std::vector<float> floats(const std::string &name, size_t count) {
        auto it = tensors.find(name);
        if(it == tensors.end()) {
            fail("Missing tensor " + name);
            return {};
        }
        std::vector<float> v = it->second.matrix.to_f32();
        if(v.size() != count) {
            fail("Unexpected size of " + name);
            return {};
        }
        return v;
    }

    bool ok() const { return error.empty(); }

private:
    TensorMap &tensors;
    std::string &error;

    const Matrix *fail(const std::string &message) {
        if(error.empty()) error = message;
        return nullptr;
    }
};

bool read_config(const Json &j, Config &c, std::string &error) {
    const Json &enc = j["encoder_config"];
    c.hidden = enc["hidden_size"].as_int();
    c.intermediate = enc["intermediate_size"].as_int();
    c.layers = enc["num_hidden_layers"].as_int();
    c.heads = enc["num_attention_heads"].as_int();
    c.conv_kernel = enc["conv_kernel_size"].as_int();
    c.sub_channels = enc["subsampling_conv_channels"].as_int();
    c.mel_bins = enc["num_mel_bins"].as_int();
    c.dec_hidden = j["decoder_hidden_size"].as_int();
    c.dec_layers = j["num_decoder_layers"].as_int();
    c.vocab_size = j["vocab_size"].as_int();
    c.blank = j["blank_token_id"].as_int();
    c.max_symbols = j["max_symbols_per_step"].is_null() ? 10 : j["max_symbols_per_step"].as_int();
    for(const Json &d : j["durations"].array) c.durations.push_back(d.as_int());

    const bool supported =
            j["model_type"].string == "parakeet_tdt"
            && enc["subsampling_factor"].as_int() == 8
            && enc["subsampling_conv_kernel_size"].as_int() == 3
            && enc["subsampling_conv_stride"].as_int() == 2
            && enc["hidden_act"].string == "silu"
            && j["hidden_act"].string == "relu"
            && !enc["scale_input"].boolean
            && !enc["attention_bias"].boolean
            && !enc["convolution_bias"].boolean
            && enc["num_key_value_heads"].as_int() == c.heads
            && c.hidden > 0 && c.heads > 0 && c.hidden % c.heads == 0
            && c.layers > 0 && c.intermediate > 0 && c.conv_kernel % 2 == 1
            && c.mel_bins % 8 == 0 && c.sub_channels > 0
            && c.dec_hidden > 0 && c.dec_layers > 0
            && c.blank >= 0 && c.blank < c.vocab_size
            && !c.durations.empty();

    if(!supported) {
        error = "Unsupported Bark model configuration";
        return false;
    }
    return true;
}

} // namespace

bool is_bark_file(const uint8_t *data, size_t size) {
    return has_bark_header(data, size);
}

Model *load(const uint8_t *data, size_t size, int n_threads, std::string &error) {
    auto model = std::make_unique<Model>();
    model->pool = std::make_unique<ThreadPool>(std::max(1, n_threads));

    Json config_json;
    if(!read_bark_file(data, size, *model->pool, model->tensors, config_json, error)) return nullptr;
    if(!read_config(config_json, model->config, error)) return nullptr;

    const Config &c = model->config;
    Model &m = *model;
    Loader ld(m.tensors, error);

    const int D = c.hidden, C = c.sub_channels, H = D / c.heads;

    m.sub0_w = ld.floats("encoder.subsampling.layers.0.weight", (size_t)C * 9);
    m.sub0_b = ld.floats("encoder.subsampling.layers.0.bias", C);
    m.sub2_w = ld.floats("encoder.subsampling.layers.2.weight", (size_t)C * 9);
    m.sub2_b = ld.floats("encoder.subsampling.layers.2.bias", C);
    m.sub3_w = ld.matrix("encoder.subsampling.layers.3.weight", C, C);
    m.sub3_b = ld.floats("encoder.subsampling.layers.3.bias", C);
    m.sub5_w = ld.floats("encoder.subsampling.layers.5.weight", (size_t)C * 9);
    m.sub5_b = ld.floats("encoder.subsampling.layers.5.bias", C);
    m.sub6_w = ld.matrix("encoder.subsampling.layers.6.weight", C, C);
    m.sub6_b = ld.floats("encoder.subsampling.layers.6.bias", C);
    m.sub_linear_w = ld.matrix("encoder.subsampling.linear.weight", D, C * (c.mel_bins / 8));
    m.sub_linear_b = ld.floats("encoder.subsampling.linear.bias", D);

    for(int i = 0; i < c.layers && ld.ok(); i++) {
        const std::string p = "encoder.layers." + std::to_string(i) + ".";
        Layer L;
        L.ff1_in = ld.matrix(p + "feed_forward1.linear1.weight", c.intermediate, D);
        L.ff1_out = ld.matrix(p + "feed_forward1.linear2.weight", D, c.intermediate);
        L.ff2_in = ld.matrix(p + "feed_forward2.linear1.weight", c.intermediate, D);
        L.ff2_out = ld.matrix(p + "feed_forward2.linear2.weight", D, c.intermediate);
        L.q = ld.matrix(p + "self_attn.q_proj.weight", D, D);
        L.k = ld.matrix(p + "self_attn.k_proj.weight", D, D);
        L.v = ld.matrix(p + "self_attn.v_proj.weight", D, D);
        L.o = ld.matrix(p + "self_attn.o_proj.weight", D, D);
        L.rel_k = ld.matrix(p + "self_attn.relative_k_proj.weight", D, D);
        L.pw1 = ld.matrix(p + "conv.pointwise_conv1.weight", 2 * D, D);
        L.pw2 = ld.matrix(p + "conv.pointwise_conv2.weight", D, D);
        L.bias_u = ld.floats(p + "self_attn.bias_u", (size_t)c.heads * H);
        L.bias_v = ld.floats(p + "self_attn.bias_v", (size_t)c.heads * H);

        const char *norms[] = {"norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"};
        std::vector<float> *norm_w[] = {&L.norm_ff1_w, &L.norm_att_w, &L.norm_conv_w, &L.norm_ff2_w, &L.norm_out_w};
        std::vector<float> *norm_b[] = {&L.norm_ff1_b, &L.norm_att_b, &L.norm_conv_b, &L.norm_ff2_b, &L.norm_out_b};
        for(int n = 0; n < 5; n++) {
            *norm_w[n] = ld.floats(p + norms[n] + ".weight", D);
            *norm_b[n] = ld.floats(p + norms[n] + ".bias", D);
        }

        const std::vector<float> dw = ld.floats(p + "conv.depthwise_conv.weight", (size_t)D * c.conv_kernel);
        const std::vector<float> bn_w = ld.floats(p + "conv.norm.weight", D);
        const std::vector<float> bn_b = ld.floats(p + "conv.norm.bias", D);
        const std::vector<float> bn_mean = ld.floats(p + "conv.norm.running_mean", D);
        const std::vector<float> bn_var = ld.floats(p + "conv.norm.running_var", D);
        if(!ld.ok()) break;

        L.dw_weight.resize((size_t)D * c.conv_kernel);
        L.dw_bias.resize(D);
        for(int ch = 0; ch < D; ch++) {
            const double a = bn_w[ch] / std::sqrt((double)bn_var[ch] + BATCH_NORM_EPS);
            for(int k = 0; k < c.conv_kernel; k++) {
                L.dw_weight[(size_t)ch * c.conv_kernel + k] = (float)(dw[(size_t)ch * c.conv_kernel + k] * a);
            }
            L.dw_bias[ch] = (float)(bn_b[ch] - bn_mean[ch] * a);
        }

        m.layers.push_back(std::move(L));
    }

    m.enc_proj_w = ld.matrix("encoder_projector.weight", c.dec_hidden, D);
    m.enc_proj_b = ld.floats("encoder_projector.bias", c.dec_hidden);

    m.embedding = ld.matrix("decoder.embedding.weight", c.vocab_size, c.dec_hidden);
    for(int l = 0; l < c.dec_layers && ld.ok(); l++) {
        const std::string s = std::to_string(l);
        m.lstm_w_ih.push_back(ld.matrix("decoder.lstm.weight_ih_l" + s, 4 * c.dec_hidden, c.dec_hidden));
        m.lstm_w_hh.push_back(ld.matrix("decoder.lstm.weight_hh_l" + s, 4 * c.dec_hidden, c.dec_hidden));
        std::vector<float> b_ih = ld.floats("decoder.lstm.bias_ih_l" + s, 4 * c.dec_hidden);
        std::vector<float> b_hh = ld.floats("decoder.lstm.bias_hh_l" + s, 4 * c.dec_hidden);
        if(!ld.ok()) break;
        for(size_t i = 0; i < b_ih.size(); i++) b_ih[i] += b_hh[i];
        m.lstm_b.push_back(std::move(b_ih));
    }
    m.dec_proj_w = ld.matrix("decoder.decoder_projector.weight", c.dec_hidden, c.dec_hidden);
    m.dec_proj_b = ld.floats("decoder.decoder_projector.bias", c.dec_hidden);

    const int joint_out = c.vocab_size + (int)c.durations.size();
    m.joint_w = ld.matrix("joint.head.weight", joint_out, c.dec_hidden);
    m.joint_b = ld.floats("joint.head.bias", joint_out);

    if(!ld.ok()) return nullptr;

    // The decoded fp32 copies above are all that is needed of these
    for(const char *name : {"encoder.subsampling.layers.0.weight", "encoder.subsampling.layers.2.weight",
                            "encoder.subsampling.layers.5.weight"}) {
        m.tensors.erase(name);
    }

    m.mel_filters = mel_filterbank(c.mel_bins);

    // torch.hann_window(400, periodic=False), centered in the 512-point frame
    m.window.assign(N_FFT, 0.0f);
    const int offset = (N_FFT - WIN_LENGTH) / 2;
    for(int i = 0; i < WIN_LENGTH; i++) {
        m.window[offset + i] = (float)(0.5 - 0.5 * std::cos(2.0 * M_PI * i / (WIN_LENGTH - 1)));
    }

    m.twiddles.resize(N_FFT / 2);
    for(int i = 0; i < N_FFT / 2; i++) {
        m.twiddles[i] = std::polar(1.0, -2.0 * M_PI * i / N_FFT);
    }

    return model.release();
}

void release(Model *model) {
    delete model;
}

size_t max_samples() {
    return MAX_SAMPLES;
}

// ---------------------------------------------------------------------------
// Features
// ---------------------------------------------------------------------------

std::vector<float> Model::log_mel(const std::vector<float> &audio, int &n_frames, int &n_valid) const {
    const int L = (int)audio.size();
    const int n_mels = config.mel_bins;
    const int n_bins = N_FFT / 2 + 1;

    std::vector<float> emphasized(L);
    if(L > 0) emphasized[0] = audio[0];
    for(int i = 1; i < L; i++) emphasized[i] = audio[i] - PREEMPHASIS * audio[i - 1];

    // torch.stft(center=True, pad_mode="constant"): N_FFT/2 zeros on each side
    n_frames = 1 + L / HOP_LENGTH;
    n_valid = L / HOP_LENGTH;

    std::vector<float> mel((size_t)n_frames * n_mels);
    pool->parallel_for(n_frames, [&](int t) {
        std::vector<std::complex<double>> buf(N_FFT);
        const int start = t * HOP_LENGTH - N_FFT / 2;
        for(int i = 0; i < N_FFT; i++) {
            const int s = start + i;
            const float v = (s >= 0 && s < L) ? emphasized[s] : 0.0f;
            buf[i] = std::complex<double>((double)v * window[i], 0.0);
        }
        fft(buf, twiddles);

        float power[N_FFT / 2 + 1];
        for(int k = 0; k < n_bins; k++) power[k] = (float)std::norm(buf[k]);

        for(int m = 0; m < n_mels; m++) {
            const float *f = &mel_filters[(size_t)m * n_bins];
            float s = 0.0f;
            for(int k = 0; k < n_bins; k++) s += f[k] * power[k];
            mel[(size_t)t * n_mels + m] = std::log(s + LOG_ZERO_GUARD);
        }
    });

    // Per-feature normalization over the valid frames; the rest become zero
    for(int m = 0; m < n_mels; m++) {
        double mean = 0.0;
        for(int t = 0; t < n_valid; t++) mean += mel[(size_t)t * n_mels + m];
        mean /= n_valid;
        double var = 0.0;
        for(int t = 0; t < n_valid; t++) {
            const double d = mel[(size_t)t * n_mels + m] - mean;
            var += d * d;
        }
        var /= (n_valid - 1);
        const double denom = std::sqrt(var) + NORM_EPSILON;
        for(int t = 0; t < n_frames; t++) {
            float &v = mel[(size_t)t * n_mels + m];
            v = t < n_valid ? (float)((v - mean) / denom) : 0.0f;
        }
    }
    return mel;
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

// Conv2d subsampling. Activations are channels-last: [time][freq][channel].
// Time steps past the valid length are zeroed after every convolution.
void Model::subsample(const std::vector<float> &mel, int n_frames, int n_valid, std::vector<float> &out, int &T) {
    const int C = config.sub_channels;
    const int F0 = config.mel_bins;

    // Layer 0: 1 -> C channels, 3x3, stride 2
    int T1 = conv_out_len(n_frames), F1 = conv_out_len(F0), v1 = conv_out_len(n_valid);
    std::vector<float> a((size_t)T1 * F1 * C);
    pool->parallel_for(T1, [&](int t) {
        for(int f = 0; f < F1; f++) {
            float *dst = &a[((size_t)t * F1 + f) * C];
            if(t >= v1) {
                std::fill(dst, dst + C, 0.0f);
                continue;
            }
            float patch[9];
            for(int kt = 0; kt < 3; kt++) {
                for(int kf = 0; kf < 3; kf++) {
                    const int it = 2 * t + kt - 1, jf = 2 * f + kf - 1;
                    patch[kt * 3 + kf] = (it >= 0 && it < n_frames && jf >= 0 && jf < F0)
                                         ? mel[(size_t)it * F0 + jf] : 0.0f;
                }
            }
            for(int c = 0; c < C; c++) {
                float s = sub0_b[c];
                for(int k = 0; k < 9; k++) s += sub0_w[(size_t)c * 9 + k] * patch[k];
                dst[c] = std::max(s, 0.0f);
            }
        }
    });

    // Depthwise 3x3 stride 2, then pointwise 1x1, then ReLU
    auto dw_pw = [&](const std::vector<float> &in, int Tin, int Fin, int vin,
                     const std::vector<float> &dw_w, const std::vector<float> &dw_b,
                     const Matrix &pw_w, const std::vector<float> &pw_b,
                     std::vector<float> &res, int &Tout, int &Fout, int &vout) {
        Tout = conv_out_len(Tin);
        Fout = conv_out_len(Fin);
        vout = conv_out_len(vin);
        std::vector<float> d((size_t)Tout * Fout * C, 0.0f);
        const int to = Tout, fo = Fout, vo = vout;
        pool->parallel_for(vo, [&](int t) {
            for(int f = 0; f < fo; f++) {
                float *dst = &d[((size_t)t * fo + f) * C];
                for(int c = 0; c < C; c++) dst[c] = dw_b[c];
                for(int kt = 0; kt < 3; kt++) {
                    const int it = 2 * t + kt - 1;
                    if(it < 0 || it >= Tin) continue;
                    for(int kf = 0; kf < 3; kf++) {
                        const int jf = 2 * f + kf - 1;
                        if(jf < 0 || jf >= Fin) continue;
                        const float *src = &in[((size_t)it * Fin + jf) * C];
                        for(int c = 0; c < C; c++) dst[c] += dw_w[(size_t)c * 9 + kt * 3 + kf] * src[c];
                    }
                }
            }
        });

        res.assign((size_t)to * fo * C, 0.0f);
        const int rows = vo * fo;
        gemm_nt(rows, C, C, d.data(), C, pw_w, res.data(), C, pw_b.data(), pool.get());
        for(size_t i = 0; i < (size_t)rows * C; i++) res[i] = std::max(res[i], 0.0f);
    };

    std::vector<float> b, c2;
    int T2, F2, v2, T3, F3, v3;
    dw_pw(a, T1, F1, v1, sub2_w, sub2_b, *sub3_w, sub3_b, b, T2, F2, v2);
    dw_pw(b, T2, F2, v2, sub5_w, sub5_b, *sub6_w, sub6_b, c2, T3, F3, v3);

    // [time][channel * freq] -> linear
    T = v3;
    std::vector<float> flat((size_t)T * C * F3);
    for(int t = 0; t < T; t++) {
        for(int f = 0; f < F3; f++) {
            for(int c = 0; c < C; c++) {
                flat[(size_t)t * C * F3 + (size_t)c * F3 + f] = c2[((size_t)t * F3 + f) * C + c];
            }
        }
    }

    out.resize((size_t)T * config.hidden);
    gemm_nt(T, config.hidden, C * F3, flat.data(), C * F3, *sub_linear_w, out.data(), config.hidden,
            sub_linear_b.data(), pool.get());
}

void Model::encoder_layer(const Layer &L, std::vector<float> &x, int T, const std::vector<float> &pos) {
    const int D = config.hidden;
    const int I = config.intermediate;
    const int H = config.heads;
    const int hd = D / H;
    const int P = 2 * T - 1;
    const size_t TD = (size_t)T * D;
    ThreadPool *tp = pool.get();

    std::vector<float> h(TD), big((size_t)T * std::max(I, 2 * D)), y(TD);

    auto feed_forward = [&](const Matrix &w_in, const Matrix &w_out,
                            const std::vector<float> &nw, const std::vector<float> &nb) {
        memcpy(h.data(), x.data(), TD * sizeof(float));
        layer_norm(h.data(), T, D, nw.data(), nb.data(), LAYER_NORM_EPS);
        gemm_nt(T, I, D, h.data(), D, w_in, big.data(), I, nullptr, tp);
        tp->parallel_for(T, [&](int t) {
            float *r = &big[(size_t)t * I];
            for(int i = 0; i < I; i++) r[i] = silu(r[i]);
        });
        gemm_nt(T, D, I, big.data(), I, w_out, y.data(), D, nullptr, tp);
        for(size_t i = 0; i < TD; i++) x[i] += 0.5f * y[i];
    };

    // Feed forward 1
    feed_forward(*L.ff1_in, *L.ff1_out, L.norm_ff1_w, L.norm_ff1_b);

    // Relative position multi-head attention
    {
        memcpy(h.data(), x.data(), TD * sizeof(float));
        layer_norm(h.data(), T, D, L.norm_att_w.data(), L.norm_att_b.data(), LAYER_NORM_EPS);

        std::vector<float> q(TD), k(TD), v(TD), r((size_t)P * D), att(TD);
        gemm_nt(T, D, D, h.data(), D, *L.q, q.data(), D, nullptr, tp);
        gemm_nt(T, D, D, h.data(), D, *L.k, k.data(), D, nullptr, tp);
        gemm_nt(T, D, D, h.data(), D, *L.v, v.data(), D, nullptr, tp);
        gemm_nt(P, D, D, pos.data(), D, *L.rel_k, r.data(), D, nullptr, tp);

        const float scaling = 1.0f / std::sqrt((float)hd);
        tp->parallel_for(H, [&](int head) {
            const int off = head * hd;
            std::vector<float> qu((size_t)T * hd), qv((size_t)T * hd), vt((size_t)hd * T);
            for(int t = 0; t < T; t++) {
                for(int d = 0; d < hd; d++) {
                    const float qq = q[(size_t)t * D + off + d];
                    qu[(size_t)t * hd + d] = qq + L.bias_u[off + d];
                    qv[(size_t)t * hd + d] = qq + L.bias_v[off + d];
                    vt[(size_t)d * T + t] = v[(size_t)t * D + off + d];
                }
            }

            std::vector<float> scores((size_t)T * T), bd((size_t)T * P);
            gemm_nt(T, T, hd, qu.data(), hd, WeightRef(k.data() + off, D), scores.data(), T, nullptr, nullptr);
            gemm_nt(T, P, hd, qv.data(), hd, WeightRef(r.data() + off, D), bd.data(), P, nullptr, nullptr);

            // Position embeddings run from T-1 down to -(T-1): relative
            // position i - j is at index T - 1 - i + j
            for(int i = 0; i < T; i++) {
                float *s = &scores[(size_t)i * T];
                const float *b = &bd[(size_t)i * P + (T - 1 - i)];
                float mx = -INFINITY;
                for(int j = 0; j < T; j++) {
                    s[j] = s[j] * scaling + b[j] * scaling;
                    mx = std::max(mx, s[j]);
                }
                float sum = 0.0f;
                for(int j = 0; j < T; j++) {
                    s[j] = std::exp(s[j] - mx);
                    sum += s[j];
                }
                const float inv = 1.0f / sum;
                for(int j = 0; j < T; j++) s[j] *= inv;
            }

            gemm_nt(T, hd, T, scores.data(), T, WeightRef(vt.data(), T), att.data() + off, D, nullptr, nullptr);
        });

        gemm_nt(T, D, D, att.data(), D, *L.o, y.data(), D, nullptr, tp);
        for(size_t i = 0; i < TD; i++) x[i] += y[i];
    }

    // Convolution module
    {
        memcpy(h.data(), x.data(), TD * sizeof(float));
        layer_norm(h.data(), T, D, L.norm_conv_w.data(), L.norm_conv_b.data(), LAYER_NORM_EPS);
        gemm_nt(T, 2 * D, D, h.data(), D, *L.pw1, big.data(), 2 * D, nullptr, tp);

        // GLU over channels, into h
        tp->parallel_for(T, [&](int t) {
            const float *g = &big[(size_t)t * 2 * D];
            float *u = &h[(size_t)t * D];
            for(int c = 0; c < D; c++) u[c] = g[c] * sigmoid(g[c + D]);
        });

        // Depthwise convolution (BatchNorm folded in), SiLU, into y
        const int K = config.conv_kernel;
        const int pad = (K - 1) / 2;
        tp->parallel_for(T, [&](int t) {
            float *dst = &y[(size_t)t * D];
            for(int c = 0; c < D; c++) dst[c] = L.dw_bias[c];
            for(int k = 0; k < K; k++) {
                const int src_t = t + k - pad;
                if(src_t < 0 || src_t >= T) continue;
                const float *src = &h[(size_t)src_t * D];
                for(int c = 0; c < D; c++) dst[c] += L.dw_weight[(size_t)c * K + k] * src[c];
            }
            for(int c = 0; c < D; c++) dst[c] = silu(dst[c]);
        });

        gemm_nt(T, D, D, y.data(), D, *L.pw2, h.data(), D, nullptr, tp);
        for(size_t i = 0; i < TD; i++) x[i] += h[i];
    }

    // Feed forward 2
    feed_forward(*L.ff2_in, *L.ff2_out, L.norm_ff2_w, L.norm_ff2_b);

    layer_norm(x.data(), T, D, L.norm_out_w.data(), L.norm_out_b.data(), LAYER_NORM_EPS);
}

bool Model::encode(const std::vector<float> &mel, int n_frames, int n_valid, const volatile int *cancel,
                   std::vector<float> &out, int &T) {
    const int D = config.hidden;

    std::vector<float> x;
    subsample(mel, n_frames, n_valid, x, T);
    if(T <= 0) return true;

    // Sinusoidal relative positions T-1 .. -(T-1), interleaved sin/cos
    const int P = 2 * T - 1;
    std::vector<float> pos((size_t)P * D);
    for(int p = 0; p < P; p++) {
        const float position = (float)(T - 1 - p);
        for(int i = 0; i < D / 2; i++) {
            const float inv_freq = 1.0f / std::pow(10000.0f, (float)(2 * i) / (float)D);
            const float angle = position * inv_freq;
            pos[(size_t)p * D + 2 * i] = std::sin(angle);
            pos[(size_t)p * D + 2 * i + 1] = std::cos(angle);
        }
    }

    for(const Layer &L : layers) {
        if(cancelled(cancel)) return false;
        encoder_layer(L, x, T, pos);
    }

    out.resize((size_t)T * config.dec_hidden);
    gemm_nt(T, config.dec_hidden, D, x.data(), D, *enc_proj_w, out.data(), config.dec_hidden,
            enc_proj_b.data(), pool.get());
    return true;
}

// ---------------------------------------------------------------------------
// TDT greedy decoding
// ---------------------------------------------------------------------------

void Model::decode(const std::vector<float> &enc, int T, int T_max, const volatile int *cancel,
                   std::vector<int> &tokens) {
    const int Hd = config.dec_hidden;
    const int V = config.vocab_size;
    const int n_dur = (int)config.durations.size();
    ThreadPool *tp = pool.get();

    std::vector<float> h((size_t)config.dec_layers * Hd, 0.0f), c((size_t)config.dec_layers * Hd, 0.0f);
    std::vector<float> dec_out(Hd), x(Hd), gates(4 * Hd), gates_hh(4 * Hd), lstm_out(Hd);

    auto predict = [&](int token) {
        embedding->dequantize(token, 0, Hd, x.data());
        for(int l = 0; l < config.dec_layers; l++) {
            float *hl = &h[(size_t)l * Hd];
            float *cl = &c[(size_t)l * Hd];
            gemv(*lstm_w_ih[l], x.data(), gates.data(), lstm_b[l].data(), tp);
            gemv(*lstm_w_hh[l], hl, gates_hh.data(), nullptr, tp);
            // PyTorch gate order: input, forget, cell, output
            for(int i = 0; i < Hd; i++) {
                const float ig = sigmoid(gates[i] + gates_hh[i]);
                const float fg = sigmoid(gates[Hd + i] + gates_hh[Hd + i]);
                const float gg = std::tanh(gates[2 * Hd + i] + gates_hh[2 * Hd + i]);
                const float og = sigmoid(gates[3 * Hd + i] + gates_hh[3 * Hd + i]);
                cl[i] = fg * cl[i] + ig * gg;
                hl[i] = og * std::tanh(cl[i]);
            }
            memcpy(x.data(), hl, Hd * sizeof(float));
        }
        gemv(*dec_proj_w, x.data(), dec_out.data(), dec_proj_b.data(), tp);
    };

    // Generation starts from the blank token with an empty LSTM state
    predict(config.blank);

    std::vector<float> joint_in(Hd), logits(V + n_dur);
    const int max_steps = config.max_symbols * T_max;
    int frame = 0, symbols_at_frame = 0;

    for(int step = 0; step < max_steps && frame < T; step++) {
        if(cancelled(cancel)) return;

        const float *e = &enc[(size_t)frame * Hd];
        for(int i = 0; i < Hd; i++) joint_in[i] = std::max(e[i] + dec_out[i], 0.0f);
        gemv(*joint_w, joint_in.data(), logits.data(), joint_b.data(), tp);

        const int token = (int)(std::max_element(logits.begin(), logits.begin() + V) - logits.begin());
        const int dur_idx = (int)(std::max_element(logits.begin() + V, logits.end()) - (logits.begin() + V));
        int duration = config.durations[dur_idx];

        const bool blank = token == config.blank;
        if(blank && duration == 0) duration = 1;

        if(!blank) {
            if(token == EOS_TOKEN) break;
            tokens.push_back(token);
            predict(token);
        }

        // Like NeMo, move on after too many symbols on one frame
        if(duration == 0) {
            if(++symbols_at_frame >= config.max_symbols) {
                duration = 1;
                symbols_at_frame = 0;
            }
        } else {
            symbols_at_frame = 0;
        }
        frame += duration;
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

bool transcribe(Model *model, const float *samples, size_t n_samples, const volatile int *cancel,
                std::vector<int> &tokens, std::string &error) {
    tokens.clear();
    if(n_samples > MAX_SAMPLES) {
        error = "Audio is too long";
        return false;
    }

    std::vector<float> audio(LEADING_SILENCE + n_samples, 0.0f);
    if(n_samples) memcpy(audio.data() + LEADING_SILENCE, samples, n_samples * sizeof(float));

    int n_frames, n_valid;
    const std::vector<float> mel = model->log_mel(audio, n_frames, n_valid);

    std::vector<float> enc;
    int T = 0;
    if(!model->encode(mel, n_frames, n_valid, cancel, enc, T)) {
        error = "Cancelled";
        return false;
    }

    // transformers bounds generation by the padded encoder length
    int T_max = n_frames;
    for(int i = 0; i < 3; i++) T_max = conv_out_len(T_max);

    model->decode(enc, T, T_max, cancel, tokens);
    if(model->cancelled(cancel)) {
        error = "Cancelled";
        return false;
    }
    return true;
}

} // namespace bark
