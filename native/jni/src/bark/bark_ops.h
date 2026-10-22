#ifndef LATINIME_BARK_OPS_H
#define LATINIME_BARK_OPS_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace bark {

// A weight matrix of rows x cols, stored in one of the formats a .bark file
// decodes to without loss
struct Matrix {
    enum class Type { F32, I8, Levels };

    Type type = Type::F32;
    int rows = 0;
    int cols = 0;

    // F32: rows x cols
    std::vector<float> f32;

    // I8: rows x cols integers, value = q * scale[row]
    std::vector<int8_t> i8;
    std::vector<float> scale;

    // Levels: each row has 2K+1 values {0, +l1..+lK, -l1..-lK}; codes index into
    // lut (rows x lut_stride) and are packed codes_per_byte to a byte, base 2K+1
    int lut_stride = 0;
    std::vector<float> lut;
    int codes_per_byte = 0;
    int row_bytes = 0;
    std::vector<uint8_t> packed;
    std::vector<uint8_t> unpack; // 256 x codes_per_byte

    // Writes row r, columns [k0, k1), as floats
    void dequantize(int r, int k0, int k1, float *out) const;

    std::vector<float> to_f32() const;
    size_t memory_bytes() const;
};

class ThreadPool {
public:
    explicit ThreadPool(int n_threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    int size() const { return (int)workers.size() + 1; }

    // Runs fn(i) for i in [0, n), spread over all threads including the caller
    void parallel_for(int n, const std::function<void(int)> &fn);

private:
    void worker_loop();
    void run_tasks();

    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable start_cv;
    std::condition_variable done_cv;
    const std::function<void(int)> *job = nullptr;
    int job_size = 0;
    std::atomic<int> next_task{0};
    int busy = 0;
    uint64_t generation = 0;
    bool stopping = false;
};

// Rows of a weight operand for gemm_nt: either a Matrix or plain floats
struct WeightRef {
    const Matrix *matrix = nullptr;
    const float *data = nullptr;
    int ld = 0;

    WeightRef(const Matrix &m) : matrix(&m) {}
    WeightRef(const float *d, int ld_) : data(d), ld(ld_) {}
};

// Y[t][n] = sum_k X[t][k] * W[n][k] (+ bias[n]) for t < T, n < N, k < K.
// With pool, the work is split over output columns; without, it runs on the
// calling thread.
void gemm_nt(int T, int N, int K, const float *X, int ldx, WeightRef W, float *Y, int ldy,
             const float *bias, ThreadPool *pool);

// y[n] = W[n] . x (+ bias[n]), single vector
void gemv(const Matrix &W, const float *x, float *y, const float *bias, ThreadPool *pool);

void layer_norm(float *x, int T, int D, const float *weight, const float *bias, float eps);

} // namespace bark

#endif //LATINIME_BARK_OPS_H
