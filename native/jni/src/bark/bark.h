#ifndef LATINIME_BARK_H
#define LATINIME_BARK_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Speech recognition for Bark (.bark) models: a compressed Parakeet TDT
// (FastConformer encoder + LSTM/TDT transducer decoder), English only.

namespace bark {

struct Model;

// Returns true if the buffer looks like a .bark file (only checks the header)
bool is_bark_file(const uint8_t *data, size_t size);

// Decodes the model from the buffer. The buffer is not referenced afterwards.
// Returns nullptr and sets error on failure.
Model *load(const uint8_t *data, size_t size, int n_threads, std::string &error);

void release(Model *model);

// Transcribes 16 kHz mono audio. Returns token ids; 0..vocab_size-1 index the
// tokenizer vocabulary. Returns false if cancelled or the audio is too long.
bool transcribe(Model *model, const float *samples, size_t n_samples, const volatile int *cancel,
                std::vector<int> &tokens, std::string &error);

// Longest clip transcribe() accepts, in samples
size_t max_samples();

} // namespace bark

#endif //LATINIME_BARK_H
