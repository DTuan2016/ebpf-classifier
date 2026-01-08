#!/usr/bin/python3
# -*- coding: utf-8 -*-


from bcc import BPF
from bcc import lib
import sys
import os
import time
import json
from socket import inet_ntop, ntohs, AF_INET, AF_INET6
from struct import pack
import ctypes as ct
import joblib
from datetime import datetime

curdir = os.path.dirname(__file__)
def usage():
    print("Usage: {0} <ifdev> <flag>".format(sys.argv[0]))
    exit(1)

bpf_text = """
#include <uapi/linux/bpf.h>
#include <linux/inet.h>
#include <linux/ip.h>
#include <uapi/linux/tcp.h>
#include <uapi/linux/udp.h>

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))
// nn_math.h
#define INT8_MAX_VALUE 127
#define FXP_VALUE 16
#define ROUND_CONST (1 << (FXP_VALUE - 1)) // = 0.5 to before right shifting to improve rounding
// mlp_params.h
#define N 1
#define INPUT_DIM 6
#define H1 16
#define H2 16
#define OUTPUT_DIM 2

struct pkt_key_t {
  u32 protocol;
  u32 saddr;
  u32 daddr;
  u32 sport;
  u32 dport;
};

struct pkt_leaf_t {
  u64 start_ts;
  u64 last_seen;
  u64 min_IAT;
  u32 total_pkts;
  u32 max_pkt_len;
  u32 min_pkt_len;
  u32 total_bytes;
};

struct accounting{
  u64 time_in;
  u64 proc_time;
  u32 total_bytes;
  u32 total_pkts;
};

BPF_HASH(accounting_map, u32, struct accounting, 1);
BPF_PERCPU_ARRAY(out_input2, int64_t, 16);
BPF_PERCPU_ARRAY(out_input, int64_t, 16);
// BPF_ARRAY(out_input2, int64_t, 16);
// BPF_ARRAY(out_input, int64_t, 16);
BPF_ARRAY(layer_1_weight, int, LAYER_1_WEIGHT);
BPF_ARRAY(layer_2_weight, int, LAYER_2_WEIGHT);
BPF_ARRAY(layer_3_weight, int, LAYER_3_WEIGHT);
BPF_ARRAY(layer_1_s_w_inv, int, LAYER_1_S_W_INV);
BPF_ARRAY(layer_2_s_w_inv, int, LAYER_2_S_W_INV);
BPF_ARRAY(layer_3_s_w_inv, int, LAYER_3_S_W_INV);
BPF_ARRAY(data_min, int64_t, DATA_MIN);
BPF_ARRAY(data_scale, int64_t, DATA_SCALE);
BPF_ARRAY(layer_1_s_x, int, 1);
BPF_ARRAY(layer_2_s_x, int, 1);
BPF_ARRAY(layer_3_s_x, int, 1);
BPF_ARRAY(layer_1_s_x_inv, int, 1);
BPF_ARRAY(layer_2_s_x_inv, int, 1);
BPF_ARRAY(layer_3_s_x_inv, int, 1);
BPF_TABLE("lru_hash", struct pkt_key_t, struct pkt_leaf_t, sessions, 1024);
BPF_TABLE("prog", int, int, jmp_table, 8);
BPF_HASH(dropcnt, int, u32);

int nn1(struct xdp_md *ctx) {
  unsigned int k, m, _k, _m;
  int _zero = 0;
  int64_t _zero64 = 0;
  int rounded_value, tensor_int, tensor_frac, scale_factor_int, scale_factor_frac, accumulator, s_w_inv, s_x, s_x_inv, out_value;
  int64_t out;
  int8_t weight, x_q[H1];
  // linear_layer(out_input, layer_2_weight, out_h1, layer_2_s_w_inv, dimension_layer2);
  // quantize(x, x_q, layer_2_s_x, layer_2_s_x_inv, N*H1);
  s_x = *((int*)layer_2_s_x.lookup_or_init(&_zero, &_zero));
  s_x_inv = *((int*)layer_2_s_x_inv.lookup_or_init(&_zero, &_zero));
  scale_factor_int = (s_x + ROUND_CONST) >> FXP_VALUE;
  scale_factor_frac = s_x - (scale_factor_int << FXP_VALUE);
  for (m = 0; m < H1; m++) {
    _m = m;
    out = *out_input.lookup_or_init(&_m, &_zero64);
    tensor_int = (out + ROUND_CONST) >> FXP_VALUE;
    if (tensor_int > INT8_MAX_VALUE*s_x_inv) {
      x_q[m] = (int8_t)INT8_MAX_VALUE;
    } else if (tensor_int < -INT8_MAX_VALUE*s_x_inv) {
      x_q[m] = -(int8_t)INT8_MAX_VALUE;
    } else {
      tensor_frac = out - (tensor_int << FXP_VALUE);
      rounded_value = tensor_int*scale_factor_frac + scale_factor_int*tensor_frac;
      rounded_value += (tensor_frac*scale_factor_frac + ROUND_CONST) >> FXP_VALUE;
      rounded_value = ((rounded_value + ROUND_CONST) >> FXP_VALUE) + tensor_int*scale_factor_int;
      x_q[m] = (int8_t)rounded_value; /* store quantized value in output tensor */
    }
  }
  // mat_mult(x_q, w, output, dimension_layer);
  for (m = 0; m < H2; m++) {
    accumulator = 0;
    for (k = 0; k < H1; k++) {
      _k = k*H2 + m;
      weight = *(int8_t*)layer_2_weight.lookup_or_init(&_k, &_zero);
      accumulator += x_q[k] * weight;
    }
    // dequantize_per_row(output, w_scale_factor_inv, layer_2_s_x_inv, N, H2);
    _m = m;
    // s_w_inv = *((int*)layer_2_s_w_inv.lookup_or_init(&_m, &_zero));
    out_value = *layer_2_s_w_inv.lookup_or_init(&_m, &_zero);
    // out_value = s_w_inv * s_x_inv;
    out = (int64_t)accumulator;
    if (out_value > (1 << FXP_VALUE)) {
      out *= ((out_value + ROUND_CONST) >> FXP_VALUE);
    }
    else {
      out = (out_value*out + ROUND_CONST) >> FXP_VALUE;
    }
    // relu(output, N*M);
    out = MAX(out, 0);
    out_input2.update(&_m, &out);
  }
  jmp_table.call(ctx, 1);
  return XDP_DROP;
}
int nn2(struct xdp_md *ctx) {
  unsigned int k, m, _k, _m;
  int _zero = 0;
  int64_t _zero64 = 0;
  int rounded_value, tensor_int, tensor_frac, scale_factor_int, scale_factor_frac, accumulator, s_w_inv, s_x, s_x_inv, out_value;
  int64_t out;
  int8_t weight, x_q[H2];
  s_x = *((int*)layer_3_s_x.lookup_or_init(&_zero, &_zero));
  s_x_inv = *layer_3_s_x_inv.lookup_or_init(&_zero, &_zero);
  scale_factor_int = (s_x + ROUND_CONST) >> FXP_VALUE;
  scale_factor_frac = s_x - (scale_factor_int << FXP_VALUE);
  for (m = 0; m < H2; m++) {
    _m = m;
    out = *out_input2.lookup_or_init(&_m, &_zero64);
    tensor_int = (out + ROUND_CONST) >> FXP_VALUE;
    if (tensor_int > INT8_MAX_VALUE*s_x_inv) {
      x_q[m] = (int8_t)INT8_MAX_VALUE;
    } else if (tensor_int < -INT8_MAX_VALUE*s_x_inv) {
      x_q[m] = -(int8_t)INT8_MAX_VALUE;
    } else {
      tensor_frac = out - (tensor_int << FXP_VALUE);
      rounded_value = tensor_int*scale_factor_frac + scale_factor_int*tensor_frac;
      rounded_value += (tensor_frac*scale_factor_frac + ROUND_CONST) >> FXP_VALUE;
      rounded_value = ((rounded_value + ROUND_CONST) >> FXP_VALUE) + tensor_int*scale_factor_int;
      x_q[m] = (int8_t)rounded_value; /* store quantized value in output tensor */
    }
  }
  // mat_mult(x_q, w, output, dimension_layer);
  unsigned int argmax_over_cols = 0;
  rounded_value = 0;
  for (m = 0; m < OUTPUT_DIM; m++) {
    accumulator = 0;
    for (k = 0; k < H2; k++) {
      _k = k*OUTPUT_DIM + m;
      weight = *(int8_t*)layer_3_weight.lookup_or_init(&_k, &_zero);
      accumulator += x_q[k] * weight;
    }
    _m = m;
    // s_w_inv = *((int*)layer_3_s_w_inv.lookup_or_init(&_m, &_zero));
    out_value = *layer_3_s_w_inv.lookup_or_init(&_m, &_zero);
    // out_value = s_w_inv * s_x_inv;
    out = (int64_t)accumulator;
    if (out_value > (1 << FXP_VALUE)) {
      out *= ((out_value + ROUND_CONST) >> FXP_VALUE);
    } else {
      out = (out_value*out + ROUND_CONST) >> FXP_VALUE;
    }
    if (out > rounded_value) {
      rounded_value = out;
      argmax_over_cols = m; // return column
    }
  }
  if (argmax_over_cols != 0) {
    // return XDP_DROP;
    // jmp_table.call(ctx, 2);
    // return XDP_PASS;
  }
  u32 val = 0, *vp;
  vp = dropcnt.lookup_or_init(&_zero, &val);
  *vp += 1;
  // jmp_table.call(ctx, 2);
  
  u32 key_ac = 0;
  struct accounting *acct = accounting_map.lookup(&key_ac);
  struct accounting zero_ac =  {};
  if(!acct){
    accounting_map.update(&key_ac, &zero_ac);
    acct = accounting_map.lookup(&key_ac);
  }
  if(acct){
    u64 time_out = bpf_ktime_get_ns();
    acct->proc_time += time_out - acct->time_in; 
  }
  return XDP_DROP;
  return XDP_PASS;
}
int nn_xdp_drop_packet(struct xdp_md *ctx) {
  u64 ts = bpf_ktime_get_ns();
  void* data_end = (void*)(long)ctx->data_end;
  void* data = (void*)(long)ctx->data;
  struct ethhdr *eth = data;
  u64 nh_off = sizeof(*eth);
  struct iphdr *iph;
  struct tcphdr *th;
  struct udphdr *uh;
  struct pkt_key_t pkt_key = {};
  int _zero = 0;
  int64_t _zero64 = 0;

  pkt_key.protocol = 0;
  pkt_key.saddr = 0;
  pkt_key.daddr = 0;
  pkt_key.sport = 0;
  pkt_key.dport = 0;

  ethernet: {
    if (data + nh_off > data_end) {
      return XDP_DROP;
    }
    switch(eth->h_proto) {
      case htons(ETH_P_IP): goto ip;
      default: return XDP_PASS;
    }
  }
  ip: {
    iph = data + nh_off;
    if ((void*)&iph[1] > data_end)
      return XDP_DROP;
    pkt_key.saddr    = iph->saddr;
    pkt_key.daddr    = iph->daddr;
    pkt_key.protocol = iph->protocol;

    switch(iph->protocol) {
      case IPPROTO_TCP: goto tcp;
      case IPPROTO_UDP: goto udp;
      default: return XDP_PASS;
    }
  }
  tcp: {
    th = (struct tcphdr *)(iph + 1);
    if ((void*)(th + 1) > data_end) {
      return XDP_DROP;
    }
    pkt_key.sport = ntohs(th->source);
    pkt_key.dport = ntohs(th->dest);

    goto nn;
  }
  udp: {
    uh = (struct udphdr *)(iph + 1);
    if ((void*)(uh + 1) > data_end) {
      return XDP_DROP;
    }
    pkt_key.sport = ntohs(uh->source);
    pkt_key.dport = ntohs(uh->dest);

    goto nn;
  }
  nn: {
    struct pkt_leaf_t *pkt_leaf = sessions.lookup(&pkt_key);
    if (!pkt_leaf) {
      u64 pkt_len_zero = ntohs(iph->tot_len);
      struct pkt_leaf_t zero = {};
      zero.start_ts = ts;
      zero.last_seen = ts;
      zero.min_IAT = 0xFFFFFFFFFFFFFFFFULL;
      zero.min_pkt_len = pkt_len_zero;
      zero.max_pkt_len = pkt_len_zero;
      zero.total_pkts = 1;
      zero.total_bytes = pkt_len_zero;
      sessions.update(&pkt_key, &zero);
      pkt_leaf = sessions.lookup(&pkt_key);
    }
    if (pkt_leaf != NULL) {
      u32 key_ac = 0;
      struct accounting *acct = accounting_map.lookup(&key_ac);
      struct accounting zero_ac =  {};
      u64 pkt_len = ntohs(iph->tot_len);
      if(!acct){
        accounting_map.update(&key_ac, &zero_ac);
        acct = accounting_map.lookup(&key_ac);
      }    
      if(acct){
        acct->time_in = bpf_ktime_get_ns();
      }
      u64 x[INPUT_DIM] = {0};
      u64 iat_ns = (pkt_leaf->last_seen > 0 && ts >= pkt_leaf->last_seen) ? (ts - pkt_leaf->last_seen) : 0;
      pkt_leaf->total_bytes += pkt_len;
      pkt_leaf->total_pkts += 1;
      
      if(iat_ns > 0 && iat_ns < pkt_leaf->min_IAT){
        pkt_leaf->min_IAT = iat_ns;
      }
      
      if(pkt_len > pkt_leaf->max_pkt_len){
        pkt_leaf->max_pkt_len = pkt_len;
      }
      
      if(pkt_len < pkt_leaf->min_pkt_len){
        pkt_leaf->min_pkt_len = pkt_len;
      }
      
      pkt_leaf->last_seen = ts;
      
      x[0] = pkt_leaf->last_seen - pkt_leaf->start_ts;
      x[1] = pkt_leaf->total_pkts;
      x[2] = pkt_leaf->total_bytes;
      x[3] = pkt_leaf->max_pkt_len;
      x[4] = pkt_leaf->min_pkt_len;
      x[5] = pkt_leaf->min_IAT;
      
      x[0] <<= FXP_VALUE;
      x[1] <<= FXP_VALUE;
      x[2] <<= FXP_VALUE;
      x[3] <<= FXP_VALUE;
      x[4] <<= FXP_VALUE;
      x[5] <<= FXP_VALUE;
      
      unsigned int k, m, _k, _m;

      sessions.update(&pkt_key, pkt_leaf);

      int64_t out, _data_scale;
      // NORMALIZATION
      for (k = 0; k < INPUT_DIM; k++) {
        unsigned int _k = k;
        out = *data_min.lookup_or_init(&_k, &_zero64);
        _data_scale = *data_scale.lookup_or_init(&_k, &_zero64);
        x[k] = (x[k] - out) * _data_scale;
      }

      int rounded_value, tensor_int, tensor_frac, scale_factor_int, scale_factor_frac, accumulator, s_w_inv, s_x, s_x_inv, out_value;
      int8_t weight, x_q[INPUT_DIM];

      // linear_layer(x, layer_1_weight, out_input, layer_1_s_w_inv, dimension_layer1);
      // quantize(x, x_q, x_scale_factor, x_scale_factor_inv, N*INPUT_DIM);
      s_x = *((int*)layer_1_s_x.lookup_or_init(&_zero, &_zero));
      s_x_inv = *((int*)layer_1_s_x_inv.lookup_or_init(&_zero, &_zero));
      scale_factor_int = (s_x + ROUND_CONST) >> FXP_VALUE;
      scale_factor_frac = s_x - (scale_factor_int << FXP_VALUE);
      for (m = 0; m < INPUT_DIM; m++) {
        tensor_int = (x[m] + ROUND_CONST) >> FXP_VALUE;
        if (tensor_int > INT8_MAX_VALUE*s_x_inv) {
          x_q[m] = (int8_t)INT8_MAX_VALUE;
        } else if (tensor_int < -INT8_MAX_VALUE*s_x_inv) {
          x_q[m] = -(int8_t)INT8_MAX_VALUE;
        } else {
          tensor_frac = x[m] - (tensor_int << FXP_VALUE);
          rounded_value = tensor_int*scale_factor_frac + scale_factor_int*tensor_frac;
          rounded_value += (tensor_frac*scale_factor_frac + ROUND_CONST) >> FXP_VALUE;
          rounded_value = ((rounded_value + ROUND_CONST) >> FXP_VALUE) + tensor_int*scale_factor_int;
          x_q[m] = (int8_t)rounded_value; /* store quantized value in output tensor */
        }
      }
      // mat_mult(x_q, w, output, dimension_layer);
      for (m = 0; m < H1; m++) {
        accumulator = 0;
        for (k = 0; k < INPUT_DIM; k++) {
          _k = k*H1 + m;
          weight = *(int8_t*)layer_1_weight.lookup_or_init(&_k, &_zero);
          accumulator += x_q[k] * weight;
        }
        // dequantize_per_row(output, w_scale_factor_inv, x_scale_factor_inv, N, M);
        _m = m;
        // s_w_inv = *((int*)layer_1_s_w_inv.lookup_or_init(&_m, &_zero));
        out_value = *layer_1_s_w_inv.lookup_or_init(&_m, &_zero);
        // out_value = s_w_inv * s_x_inv;
        out = (int64_t)accumulator;
        if (out_value > (1 << FXP_VALUE)) {
          out *= ((out_value + ROUND_CONST) >> FXP_VALUE);
        } else {
          out = (out_value*out + ROUND_CONST) >> FXP_VALUE;
        }
        // relu(output, N*H1);
        out = MAX(out, 0);
        out_input.update(&_m, &out);
      }
      if(acct){
        acct->total_bytes += pkt_len;
        acct->total_pkts += 1;
      }
      jmp_table.call(ctx, 0);
    }
  }
  return XDP_PASS;
}
"""

def map_bpf_table(hashmap, values, c_type='int'):
    MAP_SIZE = len(values)
    assert len(hashmap.items()) == MAP_SIZE
    keys = (hashmap.Key * MAP_SIZE)()
    new_values = (hashmap.Leaf * MAP_SIZE)()

    for i in range(MAP_SIZE):
        keys[i] = ct.c_int(i)
        if c_type == 'int':
            new_values[i] = ct.c_int(values[i])
        elif c_type == 'int8_t':
            new_values[i] = ct.c_char(values[i])
        elif c_type == 'longlong':
            new_values[i] = ct.c_longlong(values[i])
        else:
            new_values[i] = ct.c_longlong(values[i])
    hashmap.items_update_batch(keys, new_values)

def read_accounting(acct_map):
    key = ct.c_uint(0)
    try:
        return acct_map[key]
    except KeyError:
        # Trả về struct rỗng nếu chưa có entry
        class Accounting(ct.Structure):
            _fields_ = [
                ("time_in", ct.c_ulonglong),
                ("proc_time", ct.c_ulonglong),
                ("total_bytes", ct.c_uint),
                ("total_pkts", ct.c_uint),
            ]
        return Accounting()
      
def compute_metrics(prev, now, interval):
    delta_bytes = now.total_bytes - prev.total_bytes
    delta_pkts  = now.total_pkts  - prev.total_pkts
    delta_proc  = now.proc_time   - prev.proc_time  # ns

    # Chống counter reset
    if delta_bytes < 0: delta_bytes = 0
    if delta_pkts  < 0: delta_pkts  = 0
    if delta_proc  < 0: delta_proc  = 0

    throughput_bps = delta_bytes / interval if interval > 0 else 0
    pps            = delta_pkts  / interval if interval > 0 else 0
    avg_latency_ns = (delta_proc / delta_pkts) if delta_pkts > 0 else 0

    return throughput_bps, avg_latency_ns, pps

if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        usage()
    device = sys.argv[1]
    log_file = sys.argv[2]
    maptype = "percpu_array"
    flags = 0
    offload_device = None
    if len(sys.argv) == 4:
        if "-S" in sys.argv:
            # XDP_FLAGS_SKB_MODE
            flags |= BPF.XDP_FLAGS_SKB_MODE
        if "-D" in sys.argv:
            # XDP_FLAGS_DRV_MODE
            flags |= BPF.XDP_FLAGS_DRV_MODE
        if "-H" in sys.argv:
            # XDP_FLAGS_HW_MODE
            maptype = "array"
            offload_device = device.encode()
            flags |= BPF.XDP_FLAGS_HW_MODE
    prefix_path = "runs"
    with open(f'{curdir}/mlp_params.json') as f:
        params = json.load(f)

    bpf_text = bpf_text.replace('LAYER_1_S_W_INV', str(len(params["layer_1_s_w_inv"])))
    bpf_text = bpf_text.replace('LAYER_2_S_W_INV', str(len(params["layer_2_s_w_inv"])))
    bpf_text = bpf_text.replace('LAYER_3_S_W_INV', str(len(params["layer_3_s_w_inv"])))
    bpf_text = bpf_text.replace('LAYER_1_WEIGHT', str(len(params["layer_1_weight"])))
    bpf_text = bpf_text.replace('LAYER_2_WEIGHT', str(len(params["layer_2_weight"])))
    bpf_text = bpf_text.replace('LAYER_3_WEIGHT', str(len(params["layer_3_weight"])))
    bpf_text = bpf_text.replace('DATA_MIN', str(len(params["data_min"])))
    bpf_text = bpf_text.replace('DATA_SCALE', str(len(params["data_scale"])))
    # --- Tạo thư mục log nếu chưa có ---
    if os.path.dirname(log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    ret = []  
    # b = BPF(text=bpf_text, debug=0,  cflags=["-w", "-DMAPTYPE={maptype}"],
    b = BPF(text=bpf_text, debug=0,  cflags=["-w"],
            # allow_rlimit=True,
            device=offload_device)
    try:
        fn = b.load_func("nn_xdp_drop_packet", BPF.XDP)
        b.attach_xdp(device, fn, flags=flags)
        jmp_table = b.get_table("jmp_table")
        nn1_fn = b.load_func("nn1", BPF.XDP);
        nn2_fn = b.load_func("nn2", BPF.XDP);
        # fwd_fn = b.load_func("forward", BPF.XDP);
        jmp_table[ct.c_int(0)] = ct.c_int(nn1_fn.fd)
        jmp_table[ct.c_int(1)] = ct.c_int(nn2_fn.fd)
        # jmp_table[ct.c_int(2)] = ct.c_int(fwd_fn.fd)

        layer_1_weight  = b.get_table("layer_1_weight")
        layer_2_weight  = b.get_table("layer_2_weight")
        layer_3_weight  = b.get_table("layer_3_weight")
        layer_1_s_w_inv = b.get_table("layer_1_s_w_inv")
        layer_2_s_w_inv = b.get_table("layer_2_s_w_inv")
        layer_3_s_w_inv = b.get_table("layer_3_s_w_inv")
        layer_1_s_x     = b.get_table("layer_1_s_x")
        layer_2_s_x     = b.get_table("layer_2_s_x")
        layer_3_s_x     = b.get_table("layer_3_s_x")
        layer_1_s_x_inv = b.get_table("layer_1_s_x_inv")
        layer_2_s_x_inv = b.get_table("layer_2_s_x_inv")
        layer_3_s_x_inv = b.get_table("layer_3_s_x_inv")
        data_min        = b.get_table("data_min")
        data_scale      = b.get_table("data_scale")
        dropcnt          = b.get_table("dropcnt")

        map_bpf_table(layer_1_weight,  params['layer_1_weight'],  'int')
        map_bpf_table(layer_2_weight,  params['layer_2_weight'],  'int')
        map_bpf_table(layer_3_weight,  params['layer_3_weight'],  'int')
        map_bpf_table(layer_1_s_w_inv, params['layer_1_s_w_inv'], 'int')
        map_bpf_table(layer_2_s_w_inv, params['layer_2_s_w_inv'], 'int')
        map_bpf_table(layer_3_s_w_inv, params['layer_3_s_w_inv'], 'int')
        map_bpf_table(layer_1_s_x_inv, params['layer_1_s_x_inv'], 'int')
        map_bpf_table(layer_2_s_x_inv, params['layer_2_s_x_inv'], 'int')
        map_bpf_table(layer_3_s_x_inv, params['layer_3_s_x_inv'], 'int')
        map_bpf_table(layer_1_s_x,     params['layer_1_s_x'],     'int')
        map_bpf_table(layer_2_s_x,     params['layer_2_s_x'],     'int')
        map_bpf_table(layer_3_s_x,     params['layer_3_s_x'],     'int')
        map_bpf_table(data_min,        params['data_min'],        'int64_t')
        map_bpf_table(data_scale,      params['data_scale'],      'int64_t')
        prev = 0
        interval = 120
        start = datetime.now()
        # Các biến lưu trạng thái trước
        ret = []
        # Lấy các bảng cần thiết
        acct_map = b.get_table("accounting_map")

        # Ghi header CSV
        with open(log_file, 'w', buffering=1) as f:
            f.write("Throughput_Mbps,Avg_Latency_ns,PPS\n")

        interval_sec = 1
        duration_sec = 120
        start_time = time.time()
        end_time = start_time + duration_sec
        # Đọc giá trị đầu tiên
        prev_acc = read_accounting(acct_map)
        prev_time = time.time()

        print("Bắt đầu đo trong 120 giây...\n")

        while time.time() < end_time:
            time.sleep(interval_sec)

            now_acc = read_accounting(acct_map)
            now_time = time.time()

            throughput_bps, lat_ns, pps = compute_metrics(
                prev_acc, now_acc, now_time - prev_time
            )
            throughput_mbps = throughput_bps * 8 / 1e6  # B/s -> Mbps

            ret.append((throughput_mbps, lat_ns, pps))

            # Ghi dòng vào CSV ngay lập tức
            with open(log_file, 'a', buffering=1) as f:
                f.write(f"{throughput_mbps:.2f},{lat_ns:.2f},{pps:.1f}\n")

            prev_acc = now_acc
            prev_time = now_time

    except KeyboardInterrupt:
        print("\nĐã nhận Ctrl+C, kết thúc đo.", flush=True)
    finally:
        b.remove_xdp(device, flags)
        print(f"Kết quả đã lưu vào {log_file}", flush=True)
        # filename = log_file
        # if "-S" in sys.argv:
        #     # XDP_FLAGS_SKB_MODE
        #     filename = log_file
        # if "-D" in sys.argv:
        #     filename = log_file
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        # with open (filename, 'w') as f:
        #     for d in ret:
        #         f.write(f"{d}\n")

