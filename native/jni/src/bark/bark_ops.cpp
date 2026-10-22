#include "bark_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace bark {

typedef float v4f __attribute__((vector_size(16)));
typedef int8_t v4i8 __attribute__((vector_size(4)));

static inline v4f load4(const float *p) {
    v4f v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static inline v4f load4_i8(const int8_t *p) {
    v4i8 v;
    memcpy(&v, p, sizeof(v));
    return __builtin_convertvector(v, v4f);
}

static inline float hsum(v4f v) {
    return (v[0] + v[1]) + (v[2] + v[3]);
}

// ---------------------------------------------------------------------------
// Matrix
// ---------------------------------------------------------------------------

void Matrix::dequantize(int r, int k0, int k1, float *out) const {
    switch(type) {
        case Type::F32:
            memcpy(out, f32.data() + (size_t)r * cols + k0, sizeof(float) * (k1 - k0));
            break;

        case Type::I8: {
            const int8_t *q = i8.data() + (size_t)r * cols;
            const float s = scale[r];
            for(int k = k0; k < k1; k++) out[k - k0] = (float)q[k] * s;
            break;
        }

        case Type::Levels: {
            const uint8_t *src = packed.data() + (size_t)r * row_bytes + k0 / codes_per_byte;
            const float *l = lut.data() + (size_t)r * lut_stride;
            const int n = k1 - k0;
            int k = 0;
            if(k0 % codes_per_byte != 0) {
                const uint8_t *u = &unpack[*src++ * codes_per_byte];
                for(int j = k0 % codes_per_byte; j < codes_per_byte && k < n; j++) out[k++] = l[u[j]];
            }
            if(codes_per_byte == 3) {
                for(; k + 3 <= n; k += 3) {
                    const uint8_t *u = &unpack[*src++ * 3];
                    out[k] = l[u[0]];
                    out[k + 1] = l[u[1]];
                    out[k + 2] = l[u[2]];
                }
            } else if(codes_per_byte == 2) {
                for(; k + 2 <= n; k += 2) {
                    const uint8_t *u = &unpack[*src++ * 2];
                    out[k] = l[u[0]];
                    out[k + 1] = l[u[1]];
                }
            }
            while(k < n) {
                const uint8_t *u = &unpack[*src++ * codes_per_byte];
                for(int j = 0; j < codes_per_byte && k < n; j++) out[k++] = l[u[j]];
            }
            break;
        }
    }
}

std::vector<float> Matrix::to_f32() const {
    std::vector<float> out((size_t)rows * cols);
    for(int r = 0; r < rows; r++) dequantize(r, 0, cols, out.data() + (size_t)r * cols);
    return out;
}

size_t Matrix::memory_bytes() const {
    return f32.size() * sizeof(float) + i8.size() + scale.size() * sizeof(float)
           + lut.size() * sizeof(float) + packed.size() + unpack.size();
}

// ---------------------------------------------------------------------------
// ThreadPool
// ---------------------------------------------------------------------------

ThreadPool::ThreadPool(int n_threads) {
    for(int i = 1; i < n_threads; i++) {
        workers.emplace_back([this] { worker_loop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
    }
    start_cv.notify_all();
    for(auto &w : workers) w.join();
}

void ThreadPool::run_tasks() {
    int i;
    while((i = next_task.fetch_add(1)) < job_size) {
        (*job)(i);
    }
}

void ThreadPool::worker_loop() {
    uint64_t seen = 0;
    while(true) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            start_cv.wait(lock, [&] { return stopping || generation != seen; });
            if(stopping) return;
            seen = generation;
        }

        run_tasks();

        std::lock_guard<std::mutex> lock(mutex);
        if(--busy == 0) done_cv.notify_one();
    }
}

void ThreadPool::parallel_for(int n, const std::function<void(int)> &fn) {
    if(n <= 0) return;
    if(workers.empty() || n == 1) {
        for(int i = 0; i < n; i++) fn(i);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        job = &fn;
        job_size = n;
        next_task = 0;
        busy = (int)workers.size();
        generation++;
    }
    start_cv.notify_all();

    run_tasks();

    std::unique_lock<std::mutex> lock(mutex);
    done_cv.wait(lock, [&] { return busy == 0; });
    job = nullptr;
}

// ---------------------------------------------------------------------------
// GEMM
// ---------------------------------------------------------------------------

// Output columns per task, and the K block that weights are dequantized in.
// GEMM_KC is a multiple of every codes_per_byte so blocks start on a byte.
static constexpr int GEMM_NB = 64;
static constexpr int GEMM_KC = 192;

// out[i][j] = x[i] . w[j] over kc elements
static inline void kernel_4x4(const float *const x[4], const float *const w[4], int kc, float out[4][4]) {
    v4f acc[4][4];
    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++) acc[i][j] = v4f{0.0f, 0.0f, 0.0f, 0.0f};

    int k = 0;
    for(; k + 4 <= kc; k += 4) {
        const v4f x0 = load4(x[0] + k), x1 = load4(x[1] + k), x2 = load4(x[2] + k), x3 = load4(x[3] + k);
        for(int j = 0; j < 4; j++) {
            const v4f wj = load4(w[j] + k);
            acc[0][j] += x0 * wj;
            acc[1][j] += x1 * wj;
            acc[2][j] += x2 * wj;
            acc[3][j] += x3 * wj;
        }
    }

    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            float s = hsum(acc[i][j]);
            for(int kk = k; kk < kc; kk++) s += x[i][kk] * w[j][kk];
            out[i][j] = s;
        }
    }
}

static void gemm_block(int T, int n0, int n1, int K, const float *X, int ldx, const WeightRef &W,
                       float *Y, int ldy, const float *bias) {
    thread_local std::vector<float> wbuf;
    if(W.matrix != nullptr && wbuf.size() < (size_t)GEMM_NB * GEMM_KC) {
        wbuf.resize((size_t)GEMM_NB * GEMM_KC);
    }

    const float *wrows[GEMM_NB];

    for(int k0 = 0; k0 < K; k0 += GEMM_KC) {
        const int kc = std::min(GEMM_KC, K - k0);
        for(int n = n0; n < n1; n++) {
            if(W.matrix != nullptr) {
                float *dst = wbuf.data() + (size_t)(n - n0) * GEMM_KC;
                W.matrix->dequantize(n, k0, k0 + kc, dst);
                wrows[n - n0] = dst;
            } else {
                wrows[n - n0] = W.data + (size_t)n * W.ld + k0;
            }
        }

        for(int t = 0; t < T; t += 4) {
            const int mt = std::min(4, T - t);
            const float *xr[4];
            for(int i = 0; i < 4; i++) xr[i] = X + (size_t)(t + std::min(i, mt - 1)) * ldx + k0;

            for(int n = n0; n < n1; n += 4) {
                const int mn = std::min(4, n1 - n);
                const float *wr[4];
                for(int j = 0; j < 4; j++) wr[j] = wrows[n - n0 + std::min(j, mn - 1)];

                float out[4][4];
                kernel_4x4(xr, wr, kc, out);

                for(int i = 0; i < mt; i++) {
                    float *y = Y + (size_t)(t + i) * ldy + n;
                    if(k0 == 0) {
                        for(int j = 0; j < mn; j++) y[j] = out[i][j] + (bias ? bias[n + j] : 0.0f);
                    } else {
                        for(int j = 0; j < mn; j++) y[j] += out[i][j];
                    }
                }
            }
        }
    }
}

void gemm_nt(int T, int N, int K, const float *X, int ldx, WeightRef W, float *Y, int ldy,
             const float *bias, ThreadPool *pool) {
    if(T <= 0 || N <= 0) return;
    if(K <= 0) {
        for(int t = 0; t < T; t++)
            for(int n = 0; n < N; n++) Y[(size_t)t * ldy + n] = bias ? bias[n] : 0.0f;
        return;
    }

    const int n_blocks = (N + GEMM_NB - 1) / GEMM_NB;
    auto run = [&](int b) {
        const int n0 = b * GEMM_NB;
        gemm_block(T, n0, std::min(N, n0 + GEMM_NB), K, X, ldx, W, Y, ldy, bias);
    };

    if(pool != nullptr) {
        pool->parallel_for(n_blocks, run);
    } else {
        for(int b = 0; b < n_blocks; b++) run(b);
    }
}

// ---------------------------------------------------------------------------
// GEMV
// ---------------------------------------------------------------------------

static float dot_f32(const float *a, const float *b, int n) {
    v4f acc0{0, 0, 0, 0}, acc1{0, 0, 0, 0};
    int k = 0;
    for(; k + 8 <= n; k += 8) {
        acc0 += load4(a + k) * load4(b + k);
        acc1 += load4(a + k + 4) * load4(b + k + 4);
    }
    float s = hsum(acc0 + acc1);
    for(; k < n; k++) s += a[k] * b[k];
    return s;
}

static float dot_i8(const int8_t *q, const float *x, int n) {
    v4f acc0{0, 0, 0, 0}, acc1{0, 0, 0, 0};
    int k = 0;
    for(; k + 8 <= n; k += 8) {
        acc0 += load4_i8(q + k) * load4(x + k);
        acc1 += load4_i8(q + k + 4) * load4(x + k + 4);
    }
    float s = hsum(acc0 + acc1);
    for(; k < n; k++) s += (float)q[k] * x[k];
    return s;
}

void gemv(const Matrix &W, const float *x, float *y, const float *bias, ThreadPool *pool) {
    constexpr int ROWS_PER_TASK = 256;
    const int n_tasks = (W.rows + ROWS_PER_TASK - 1) / ROWS_PER_TASK;

    auto run = [&](int task) {
        const int r0 = task * ROWS_PER_TASK;
        const int r1 = std::min(W.rows, r0 + ROWS_PER_TASK);
        thread_local std::vector<float> row;

        for(int r = r0; r < r1; r++) {
            float v;
            if(W.type == Matrix::Type::I8) {
                v = dot_i8(W.i8.data() + (size_t)r * W.cols, x, W.cols) * W.scale[r];
            } else if(W.type == Matrix::Type::F32) {
                v = dot_f32(W.f32.data() + (size_t)r * W.cols, x, W.cols);
            } else {
                row.resize(W.cols);
                W.dequantize(r, 0, W.cols, row.data());
                v = dot_f32(row.data(), x, W.cols);
            }
            y[r] = v + (bias ? bias[r] : 0.0f);
        }
    };

    if(pool != nullptr) {
        pool->parallel_for(n_tasks, run);
    } else {
        for(int t = 0; t < n_tasks; t++) run(t);
    }
}

// ---------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------

void layer_norm(float *x, int T, int D, const float *weight, const float *bias, float eps) {
    for(int t = 0; t < T; t++) {
        float *row = x + (size_t)t * D;
        double mean = 0.0;
        for(int d = 0; d < D; d++) mean += row[d];
        mean /= D;
        double var = 0.0;
        for(int d = 0; d < D; d++) {
            const double v = row[d] - mean;
            var += v * v;
        }
        var /= D;
        const float inv = (float)(1.0 / std::sqrt(var + eps));
        const float m = (float)mean;
        for(int d = 0; d < D; d++) row[d] = (row[d] - m) * inv * weight[d] + bias[d];
    }
}

} // namespace bark
