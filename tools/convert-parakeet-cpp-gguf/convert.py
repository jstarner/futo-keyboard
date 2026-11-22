#!/usr/bin/env python3
"""Convert a parakeet.cpp Parakeet TDT GGUF to the layout transcribe.cpp loads.

FluidVoice publishes Fluid Speech Mini and Pro as GGUF files for parakeet.cpp
(https://github.com/mudler/parakeet.cpp). They hold Parakeet TDT 0.6B v3 weights
under NeMo's tensor names and parakeet.cpp's metadata. This script renames the
tensors and writes transcribe.cpp's metadata (as its convert-parakeet.py does for
Parakeet TDT 0.6B v3), keeping every weight's bytes.

Quantized pointwise convs stay quantized, which needs
native/transcribe/patches/0001-quantized-pointwise-weights.patch (the app build
applies it). Pass --f16-pointwise for an unpatched transcribe.cpp.

usage: convert.py <in.gguf> <out.gguf> [--name NAME] [--languages en,de,...] [--f16-pointwise]
requires: pip install gguf numpy
"""
import argparse
import hashlib

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter
from gguf.quants import dequantize

# Parakeet TDT 0.6B v3's 8192 SentencePiece pieces, joined with "\n". The score
# rule below is only known to match transcribe.cpp's converter for this tokenizer.
V3_TOKENIZER_SHA256 = '7c21c04ff6ff2f806ca43a8420479091479c805d2b7952825ae2acc4fec9424d'
V3_FIRST_SCORED_PIECE = 274  # pieces before this are special tokens with score 0

ap = argparse.ArgumentParser()
ap.add_argument('input')
ap.add_argument('output')
ap.add_argument('--name', help='display name (default: the input\'s general.name)')
ap.add_argument('--languages', default='en', help='comma-separated language codes the model is for')
ap.add_argument('--f16-pointwise', action='store_true', help='dequantize pointwise convs for unpatched transcribe.cpp')
args = ap.parse_args()

reader = GGUFReader(args.input)
fields = reader.fields


def kv(key):
    f = fields[key]
    if f.types[0] == GGUFValueType.STRING:
        return bytes(f.parts[-1]).decode()
    if f.types[0] == GGUFValueType.ARRAY:
        if f.types[1] == GGUFValueType.STRING:
            return [bytes(f.parts[i]).decode() for i in f.data]
        return [f.parts[i][0].item() for i in f.data]
    return f.parts[-1][0].item()


if kv('general.architecture') != 'parakeet' or kv('parakeet.arch') != 'tdt':
    raise SystemExit('not a parakeet.cpp Parakeet TDT model')

pieces = kv('parakeet.tokenizer.pieces')
if hashlib.sha256('\n'.join(pieces).encode()).hexdigest() != V3_TOKENIZER_SHA256:
    raise SystemExit('only models with the Parakeet TDT 0.6B v3 tokenizer are supported')

durations = kv('parakeet.tdt.durations')
languages = args.languages.split(',')
n_layers = kv('parakeet.encoder.n_layers')
n_pred_layers = kv('parakeet.decoder.pred_rnn_layers')

w = GGUFWriter(args.output, 'parakeet')
w.add_string('general.name', args.name or kv('general.name'))
w.add_array('general.languages', languages)
w.add_string('general.license', 'cc-by-4.0')
w.add_string('stt.variant', 'tdt-0.6b-v3')
w.add_bool('stt.capability.lang_detect', len(languages) > 1)
w.add_string('stt.parakeet.head_kind', 'tdt')

w.add_string('tokenizer.ggml.model', 'bpe')
w.add_uint32('tokenizer.ggml.unknown_token_id', 0)
w.add_uint32('tokenizer.ggml.blank_token_id', kv('parakeet.blank_id'))
w.add_array('tokenizer.ggml.tokens', pieces + ['<blank>'])
w.add_array('tokenizer.ggml.scores',
            [0.0 if i < V3_FIRST_SCORED_PIECE else float(V3_FIRST_SCORED_PIECE - i) for i in range(len(pieces))] + [0.0])
w.add_array('tokenizer.ggml.token_type', [2] + [1] * (len(pieces) - 1) + [3])

w.add_uint32('stt.parakeet.encoder.n_layers', n_layers)
w.add_uint32('stt.parakeet.encoder.d_model', kv('parakeet.encoder.d_model'))
w.add_uint32('stt.parakeet.encoder.n_heads', kv('parakeet.encoder.n_heads'))
w.add_uint32('stt.parakeet.encoder.d_ff', kv('parakeet.encoder.ff_dim'))
w.add_uint32('stt.parakeet.encoder.conv_kernel', kv('parakeet.encoder.conv_kernel'))
w.add_uint32('stt.parakeet.encoder.subsampling_factor', kv('parakeet.encoder.subsampling_factor'))
w.add_uint32('stt.parakeet.encoder.subsampling_channels', kv('parakeet.encoder.subsampling_conv_channels'))
w.add_uint32('stt.parakeet.encoder.pos_emb_max_len', kv('parakeet.encoder.pos_emb_max_len'))
w.add_bool('stt.parakeet.encoder.use_bias', bool(kv('parakeet.encoder.use_bias')))
w.add_bool('stt.parakeet.encoder.xscaling', bool(kv('parakeet.encoder.xscaling')))
# Full-context attention and symmetric conv padding, as in Parakeet TDT 0.6B v3
w.add_int32('stt.parakeet.encoder.att_context_left', -1)
w.add_int32('stt.parakeet.encoder.att_context_right', -1)
w.add_string('stt.parakeet.encoder.att_context_style', 'regular')
w.add_int32('stt.parakeet.encoder.conv_context_left', (kv('parakeet.encoder.conv_kernel') - 1) // 2)
w.add_int32('stt.parakeet.encoder.conv_context_right', (kv('parakeet.encoder.conv_kernel') - 1) // 2)
w.add_string('stt.parakeet.encoder.conv_norm_type', kv('parakeet.encoder.conv_norm_type'))
w.add_uint32('stt.parakeet.predictor.hidden', kv('parakeet.decoder.pred_hidden'))
w.add_uint32('stt.parakeet.predictor.n_layers', n_pred_layers)
w.add_uint32('stt.parakeet.predictor.vocab', kv('parakeet.vocab_size') + 1)
w.add_uint32('stt.parakeet.joint.hidden', kv('parakeet.joint.joint_hidden'))
w.add_uint32('stt.parakeet.joint.num_extra_outputs', len(durations))
w.add_string('stt.parakeet.joint.activation', kv('parakeet.joint.activation'))
w.add_array('stt.parakeet.tdt.durations', durations)
w.add_uint32('stt.parakeet.tdt.max_symbols', kv('parakeet.decoding.max_symbols'))

w.add_string('stt.frontend.type', 'mel')
w.add_uint32('stt.frontend.num_mels', kv('parakeet.preprocessor.n_mels'))
w.add_uint32('stt.frontend.sample_rate', kv('parakeet.preprocessor.sample_rate'))
w.add_uint32('stt.frontend.n_fft', kv('parakeet.preprocessor.n_fft'))
w.add_uint32('stt.frontend.win_length', kv('parakeet.preprocessor.win_length'))
w.add_uint32('stt.frontend.hop_length', kv('parakeet.preprocessor.hop_length'))
w.add_string('stt.frontend.window', 'hann')
w.add_string('stt.frontend.normalize', kv('parakeet.preprocessor.normalize'))
w.add_float32('stt.frontend.dither', 1e-5)
w.add_float32('stt.frontend.pre_emphasis', kv('parakeet.preprocessor.preemph'))
w.add_float32('stt.frontend.f_min', 0.0)
w.add_float32('stt.frontend.f_max', kv('parakeet.preprocessor.sample_rate') / 2)

# ---- tensors ----
tensors = {t.name: t for t in reader.tensors}
used = set()
FLOAT_TYPES = (GGMLQuantizationType.F32, GGMLQuantizationType.F16)


def copy(src, dst):
    t = tensors[src]
    used.add(src)
    if t.tensor_type in FLOAT_TYPES:
        w.add_tensor(dst, np.asarray(t.data).reshape(list(reversed([int(d) for d in t.shape]))))
    else:
        # Byte shape: gguf-py derives the element shape from it
        w.add_tensor(dst, np.asarray(t.data), raw_dtype=t.tensor_type)


def copy_pointwise(src, dst):
    t = tensors[src]
    if t.tensor_type in FLOAT_TYPES or args.f16_pointwise:
        # transcribe.cpp's conv layout: ne [1, in, out]
        used.add(src)
        n_in, n_out = int(t.shape[0]), int(t.shape[1])
        data = np.asarray(t.data)
        if t.tensor_type not in FLOAT_TYPES:
            data = dequantize(data, t.tensor_type)
        w.add_tensor(dst, data.astype(np.float16).reshape(n_out, n_in, 1))
    else:
        copy(src, dst)  # 2D [in, out], needs the patch


for i in (0, 2, 3, 5, 6):
    copy(f'encoder.pre_encode.conv.{i}.weight', f'enc.pre_encode.conv.{i}.weight')
    copy(f'encoder.pre_encode.conv.{i}.bias', f'enc.pre_encode.conv.{i}.bias')
copy('encoder.pre_encode.out.weight', 'enc.pre_encode.out.weight')
copy('encoder.pre_encode.out.bias', 'enc.pre_encode.out.bias')

BLOCK = [
    ('norm_feed_forward1.weight', 'norm_ff1.weight'), ('norm_feed_forward1.bias', 'norm_ff1.bias'),
    ('feed_forward1.linear1.weight', 'ff1.linear1.weight'), ('feed_forward1.linear2.weight', 'ff1.linear2.weight'),
    ('norm_self_att.weight', 'norm_attn.weight'), ('norm_self_att.bias', 'norm_attn.bias'),
    ('self_attn.linear_q.weight', 'attn.linear_q.weight'), ('self_attn.linear_k.weight', 'attn.linear_k.weight'),
    ('self_attn.linear_v.weight', 'attn.linear_v.weight'), ('self_attn.linear_out.weight', 'attn.linear_out.weight'),
    ('self_attn.linear_pos.weight', 'attn.linear_pos.weight'),
    ('self_attn.pos_bias_u', 'attn.pos_bias_u'), ('self_attn.pos_bias_v', 'attn.pos_bias_v'),
    ('norm_conv.weight', 'norm_conv.weight'), ('norm_conv.bias', 'norm_conv.bias'),
    ('conv.depthwise_conv.weight', 'conv.depthwise.weight'),
    ('conv.batch_norm.weight', 'conv.bn.weight'), ('conv.batch_norm.bias', 'conv.bn.bias'),
    ('conv.batch_norm.running_mean', 'conv.bn.running_mean'), ('conv.batch_norm.running_var', 'conv.bn.running_var'),
    ('norm_feed_forward2.weight', 'norm_ff2.weight'), ('norm_feed_forward2.bias', 'norm_ff2.bias'),
    ('feed_forward2.linear1.weight', 'ff2.linear1.weight'), ('feed_forward2.linear2.weight', 'ff2.linear2.weight'),
    ('norm_out.weight', 'norm_out.weight'), ('norm_out.bias', 'norm_out.bias'),
]
for i in range(n_layers):
    for s, d in BLOCK:
        copy(f'encoder.layers.{i}.{s}', f'enc.blocks.{i}.{d}')
    copy_pointwise(f'encoder.layers.{i}.conv.pointwise_conv1.weight', f'enc.blocks.{i}.conv.pointwise1.weight')
    copy_pointwise(f'encoder.layers.{i}.conv.pointwise_conv2.weight', f'enc.blocks.{i}.conv.pointwise2.weight')

copy('decoder.prediction.embed.weight', 'pred.embed.weight')
for i in range(n_pred_layers):
    copy(f'decoder.prediction.dec_rnn.lstm.weight_ih_l{i}', f'pred.lstm.{i}.Wx')
    copy(f'decoder.prediction.dec_rnn.lstm.weight_hh_l{i}', f'pred.lstm.{i}.Wh')
    # transcribe.cpp takes one LSTM bias, PyTorch has two
    b_ih = f'decoder.prediction.dec_rnn.lstm.bias_ih_l{i}'
    b_hh = f'decoder.prediction.dec_rnn.lstm.bias_hh_l{i}'
    w.add_tensor(f'pred.lstm.{i}.bias',
                 (np.asarray(tensors[b_ih].data) + np.asarray(tensors[b_hh].data)).astype(np.float32))
    used.update((b_ih, b_hh))

for s, d in [('joint.enc.weight', 'joint.enc.weight'), ('joint.enc.bias', 'joint.enc.bias'),
             ('joint.pred.weight', 'joint.pred.weight'), ('joint.pred.bias', 'joint.pred.bias'),
             ('joint.joint_net.2.weight', 'joint.out.weight'), ('joint.joint_net.2.bias', 'joint.out.bias')]:
    copy(s, d)

# transcribe.cpp builds the mel filterbank and window itself
left = sorted(set(tensors) - used - {'preprocessor.featurizer.fb', 'preprocessor.featurizer.window'})
if left:
    raise SystemExit(f'unconverted tensors: {left}')

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f'wrote {args.output} ({len(used)} source tensors)')
