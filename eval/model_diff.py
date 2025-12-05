import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from exllamav3.util.file import disk_lru_cache
from exllamav3.util.progress import ProgressBar
from exllamav3.util.memory import free_mem
from exllamav3.util.measures import cosine_error, sqnr
from exllamav3 import Config, Model, Tokenizer
from exllamav3.loader import SafetensorsCollection, VariantSafetensorsCollection
from datasets import load_dataset
import torch
import torch.nn.functional as F
import math
import yaml
from safetensors.torch import save_file

def save_tensor(tensor, path: str, tensor_name: str = None):
    if isinstance(tensor, dict):
        save_file({
            k: v for k, v in tensor.items()
        }, path)
    elif isinstance(tensor, list):
        save_file({
            f"tensor.{i}": t for i, t in enumerate(tensor)
        }, path)
    else:
        save_file({
            tensor_name or f"tensor": tensor
        }, path)


@disk_lru_cache("get_dataset_text")
def get_dataset_text(spec: dict):
    assert spec["dataset"] == "wiki2", "Only wiki2 implemented atm"
    dataset_text = "\n\n".join(
        load_dataset("wikitext", "wikitext-2-raw-v1", split = "test")
        ["text"]
    )
    return dataset_text


def get_test_tokens(tokenizer, rows, eval_len = 2048, eval_stride = 512):
    with ProgressBar("Tokenizing", rows) as pb:
        dataset_spec = { "dataset": "wiki2" }
        eval_tokens = tokenizer.encode(get_dataset_text(dataset_spec))
        num_tokens = eval_tokens.shape[-1]
        seqs = []
        for a in range(0, num_tokens - eval_len, eval_stride):
            b = a + eval_len
            seqs.append(eval_tokens[:, a:b])
            pb.update(len(seqs))
            if len(seqs) >= rows:
                break
    return torch.cat(seqs, dim = 0)[:, :]


def ppl(input_ids_, logits_):
    logprob_sum_ = 0.0
    logprob_count_ = 0
    chunksize = logits_.shape[1] * 10240 // logits_.shape[1]
    b_ = 0
    while b_ < logits_.shape[1]:
        a_ = b_
        b_ = min(b_ + chunksize, logits_.shape[1])
        logits_f = logits_[a_:b_, :].float() + 1e-10
        target_ids = input_ids_[a_ + 1:b_ + 1].to(logits_.device)
        log_probs = F.log_softmax(logits_f, dim = -1)
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        logprob_sum_ += token_log_probs.sum().item()
        logprob_count_ += target_ids.numel()
    return logprob_sum_, logprob_count_


def print_summary(metrics, mode_name, top_n=10):
    """Print summary statistics for cumulative or isolated mode metrics."""
    if not metrics:
        print(f"\n=== {mode_name} MODE SUMMARY ===")
        print("No metrics collected.")
        return

    # Sort by rfn_error (descending)
    sorted_metrics = sorted(metrics, key=lambda x: x.get('rfn_error', 0), reverse=True)

    print(f"\n=== {mode_name} MODE SUMMARY ===")
    print(f"Total layers analyzed: {len(metrics)}")

    # Top N worst layers
    print(f"\nTOP {min(top_n, len(sorted_metrics))} WORST LAYERS (by RFN error):")
    for i, m in enumerate(sorted_metrics[:top_n]):
        rfn = m.get('rfn_error', 0)
        sqnr_val = m.get('sqnr', 0)
        cos = m.get('cos_error', 0)
        layer_idx = m.get('layer_idx', -1)
        module_key = m.get('module_key', 'unknown')
        print(f"  {i+1:2d}. Layer {layer_idx:3d} [{module_key:40s}]   "
              f"rfn_err: {rfn:.6f}  sqnr: {sqnr_val:7.3f}  cos_err: {cos:.6f}")

    # Statistics
    rfn_errors = [m.get('rfn_error', 0) for m in metrics]
    sqnr_values = [m.get('sqnr', 0) for m in metrics]
    cos_errors = [m.get('cos_error', 0) for m in metrics]

    import statistics
    print(f"\nStatistics:")
    print(f"  Mean RFN error:    {statistics.mean(rfn_errors):.6f}")
    print(f"  Median RFN error:  {statistics.median(rfn_errors):.6f}")
    print(f"  Std dev:           {statistics.stdev(rfn_errors) if len(rfn_errors) > 1 else 0:.6f}")
    print(f"  Max RFN error:     {max(rfn_errors):.6f} (Layer {sorted_metrics[0]['layer_idx']})")
    print(f"  Min RFN error:     {min(rfn_errors):.6f}")
    print(f"")
    print(f"  Mean SQNR:         {statistics.mean(sqnr_values):7.3f} dB")
    print(f"  Mean Cosine error: {statistics.mean(cos_errors):.6f}")


def save_metrics_json(cumulative_metrics, isolated_metrics, logits_metrics, args, output_path):
    """Save all metrics to JSON file."""
    import json
    import statistics
    import os

    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    def compute_summary(metrics):
        if not metrics:
            return {}
        rfn_errors = [m.get('rfn_error', 0) for m in metrics]
        sqnr_values = [m.get('sqnr', 0) for m in metrics]
        cos_errors = [m.get('cos_error', 0) for m in metrics]
        sorted_metrics = sorted(metrics, key=lambda x: x.get('rfn_error', 0), reverse=True)

        return {
            "mean_rfn_error": statistics.mean(rfn_errors) if rfn_errors else 0,
            "median_rfn_error": statistics.median(rfn_errors) if rfn_errors else 0,
            "std_rfn_error": statistics.stdev(rfn_errors) if len(rfn_errors) > 1 else 0,
            "max_rfn_error": max(rfn_errors) if rfn_errors else 0,
            "max_rfn_error_layer": sorted_metrics[0]['layer_idx'] if sorted_metrics else -1,
            "min_rfn_error": min(rfn_errors) if rfn_errors else 0,
            "mean_sqnr": statistics.mean(sqnr_values) if sqnr_values else 0,
            "mean_cos_error": statistics.mean(cos_errors) if cos_errors else 0
        }

    data = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "config": {
            "rows": args.rows,
            "batch_size": args.batch_size,
            "analysis_mode": args.analysis_mode
        },
        "cumulative": {
            "layers": cumulative_metrics,
            "summary": compute_summary(cumulative_metrics),
            "top_worst": sorted(cumulative_metrics, key=lambda x: x.get('rfn_error', 0), reverse=True)[:args.top_n_worst]
        } if cumulative_metrics else {},
        "isolated": {
            "layers": isolated_metrics,
            "summary": compute_summary(isolated_metrics),
            "top_worst": sorted(isolated_metrics, key=lambda x: x.get('rfn_error', 0), reverse=True)[:args.top_n_worst]
        } if isolated_metrics else {},
        "logits": logits_metrics if logits_metrics else {}
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n -- Metrics saved to: {output_path}")


def plot_metrics(cumulative_metrics, isolated_metrics, output_path="layer_metrics_plot.png"):
    """Generate matplotlib plot of per-layer metrics."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print(" -- Warning: matplotlib not available, skipping plot generation")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Layer Quantization Error Metrics', fontsize=16)

    # Plot cumulative metrics
    if cumulative_metrics:
        cum_layers = [m['layer_idx'] for m in cumulative_metrics]
        cum_rfn = [m['rfn_error'] for m in cumulative_metrics]
        cum_sqnr = [m['sqnr'] for m in cumulative_metrics]
        cum_cos = [m['cos_error'] for m in cumulative_metrics]

        axes[0, 0].plot(cum_layers, cum_rfn, 'b-', label='Cumulative', linewidth=1.5)
        axes[0, 0].set_ylabel('RFN Error')
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_title('Relative Frobenius Norm Error')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(cum_layers, cum_sqnr, 'b-', label='Cumulative', linewidth=1.5)
        axes[0, 1].set_ylabel('SQNR (dB)')
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_title('Signal-to-Quantization-Noise Ratio')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(cum_layers, cum_cos, 'b-', label='Cumulative', linewidth=1.5)
        axes[1, 0].set_ylabel('Cosine Error')
        axes[1, 0].set_xlabel('Layer Index')
        axes[1, 0].set_title('Cosine Error')
        axes[1, 0].grid(True, alpha=0.3)

    # Plot isolated metrics
    if isolated_metrics:
        iso_layers = [m['layer_idx'] for m in isolated_metrics]
        iso_rfn = [m['rfn_error'] for m in isolated_metrics]
        iso_sqnr = [m['sqnr'] for m in isolated_metrics]
        iso_cos = [m['cos_error'] for m in isolated_metrics]

        axes[0, 0].plot(iso_layers, iso_rfn, 'r--', label='Isolated', linewidth=1.5)
        axes[0, 1].plot(iso_layers, iso_sqnr, 'r--', label='Isolated', linewidth=1.5)
        axes[1, 0].plot(iso_layers, iso_cos, 'r--', label='Isolated', linewidth=1.5)

    # Add legends
    if cumulative_metrics and isolated_metrics:
        axes[0, 0].legend()
        axes[0, 1].legend()
        axes[1, 0].legend()

    # Comparison plot (cumulative vs isolated)
    if cumulative_metrics and isolated_metrics:
        # Match layers
        cum_dict = {m['layer_idx']: m['rfn_error'] for m in cumulative_metrics}
        iso_dict = {m['layer_idx']: m['rfn_error'] for m in isolated_metrics}
        common_layers = sorted(set(cum_dict.keys()) & set(iso_dict.keys()))

        if common_layers:
            cum_rfn_common = [cum_dict[l] for l in common_layers]
            iso_rfn_common = [iso_dict[l] for l in common_layers]
            axes[1, 1].scatter(iso_rfn_common, cum_rfn_common, alpha=0.6, s=20)
            axes[1, 1].plot([0, max(max(iso_rfn_common), max(cum_rfn_common))],
                           [0, max(max(iso_rfn_common), max(cum_rfn_common))],
                           'k--', alpha=0.3, label='y=x')
            axes[1, 1].set_xlabel('Isolated RFN Error')
            axes[1, 1].set_ylabel('Cumulative RFN Error')
            axes[1, 1].set_title('Cumulative vs Isolated Error')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'Comparison requires\nboth modes',
                        ha='center', va='center', fontsize=12)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f" -- Plot saved to: {output_path}")


@torch.inference_mode()
def run_isolated_pass(model_a, model_b, config_a, config_b, all_eval_ids, device, args):
    """
    Run isolated mode analysis: measure isolated impact of each quantized layer.

    Streaming approach:
    For each layer N, use FP16 outputs from previous iteration as inputs,
    then compare FP16_LayerN vs Quant_LayerN outputs.
    """
    isolated_metrics = []

    print("\n=== Isolated Mode: Streaming layer-by-layer analysis ===")

    # Start with initial embeddings - compute them once
    print(" -- Computing initial embeddings...")
    embed_module = model_a.modules[0]
    config_a.stc.begin_deferred_load()
    embed_module.load(device if not embed_module.caps.get("prefer_cpu") else "cpu")
    config_a.stc.end_deferred_load()

    fp16_activations = []
    for batch in all_eval_ids:
        params = {}
        state = embed_module.prepare_for_device(batch, params)
        state = embed_module.forward(state, params)
        fp16_activations.append(state.cpu().clone())

    embed_module.unload()
    config_a.stc.close()
    free_mem()

    # Process each transformer layer
    for idx in range(1, len(model_a.modules) - 1):
        module_a = model_a.modules[idx]
        module_b = model_b.modules[idx]

        # Load both FP16 and Quant versions of current layer
        config_a.stc.begin_deferred_load()
        module_a.load(device if not module_a.caps.get("prefer_cpu") else "cpu")
        config_a.stc.end_deferred_load()

        config_b.stc.begin_deferred_load()
        module_b.load(device if not module_b.caps.get("prefer_cpu") else "cpu")
        config_b.stc.end_deferred_load()

        # Compare FP16 vs Quant on same FP16 inputs
        rfn_error_sum = 0
        cos_error_sum = 0
        sqnr_sum = 0
        max_diff = 0
        num_samples = 0

        # Store FP16 outputs for next iteration
        next_fp16_activations = []

        for b in range(len(fp16_activations)):
            # Use FP16 activations as input for both versions
            input_state = fp16_activations[b].to(device)

            # FP16 forward
            params_a = {}
            state_a_out = module_a.prepare_for_device(input_state.clone(), params_a)
            state_a_out = module_a.forward(state_a_out, params_a)

            # Quant forward
            params_b = {}
            state_b_out = module_b.prepare_for_device(input_state.clone(), params_b)
            state_b_out = module_b.forward(state_b_out, params_b)

            # Compute metrics
            rows = state_a_out.shape[0]
            for j in range(rows):
                sa = state_a_out[j].to(float)
                sb = state_b_out[j].to(float)
                cos_error_sum += cosine_error(sa, sb)
                sqnr_sum += sqnr(sa, sb)
                sa_diff = sa - sb
                rfn_error_sum += (torch.linalg.norm(sa_diff, 'fro') / torch.linalg.norm(sb, 'fro').mean()).item()
                sa_diff.abs_()
                md = ((sa_diff.max().item()) / torch.linalg.norm(sb, 'fro').mean()).item()
                max_diff = max(max_diff, md)
                num_samples += 1

            # Save FP16 output for next iteration (move to CPU to save VRAM)
            next_fp16_activations.append(state_a_out.cpu().clone())

        # Save metrics
        isolated_metrics.append({
            'layer_idx': idx,
            'module_key': module_a.key,
            'rfn_error': rfn_error_sum / num_samples,
            'max_diff': max_diff,
            'sqnr': sqnr_sum / num_samples,
            'cos_error': cos_error_sum / num_samples
        })

        # Unload modules
        module_a.unload()
        config_a.stc.close()
        module_b.unload()
        config_b.stc.close()
        free_mem()

        # Print progress
        print(f" -- Layer {idx:3d}  [{module_a.key:40}]  (isolated)  "
              f"rfn_err: {isolated_metrics[-1]['rfn_error']:.6f}  "
              f"sqnr: {isolated_metrics[-1]['sqnr']:9.6f}")

        # Update activations for next iteration
        fp16_activations = next_fp16_activations

    return isolated_metrics


@torch.inference_mode()
def main(args):

    device = torch.device(args.device)

    config_a = Config.from_directory(args.model_a)
    config_a.override_dynamic_seq_len(2048)
    tokenizer = Tokenizer.from_config(config_a)
    model_a = Model.from_config(config_a)

    config_b = Config.from_directory(args.model_b)
    config_b.override_dynamic_seq_len(2048)
    model_b = Model.from_config(config_b)

    # Override tensors
    if args.override:
        with open(args.override, "r") as f:
            comp = yaml.safe_load(f)
        sources = {s["id"]: s["model_dir"] for s in comp["sources"]}
        overrides = {o["key"]: sources[o["source"]] for o in comp["overrides"]}
        collections = {}
        for o_key, o_dir in overrides.items():
            if o_dir not in collections:
                collections[o_dir] = []
            collections[o_dir].append(o_key)
        if len(collections):
            vstc = VariantSafetensorsCollection(config_a.stc)
            for o_dir, o_keys in collections.items():
                print(f" -- Overriding from: {o_dir}:")
                for o_key in o_keys:
                    print(f"      {o_key}")
                vstc.add_stc(o_keys, SafetensorsCollection(o_dir))
            config_a.stc = vstc

    # Dataset
    all_eval_ids = get_test_tokens(tokenizer, args.rows)

    # Inputs
    states_a = list(all_eval_ids.split(args.batch_size))
    states_b = list(all_eval_ids.split(args.batch_size))
    all_eval_ids = list(all_eval_ids.split(args.batch_size))

    # Save input IDs
    if args.save_input_ids:
        print(f" -- Saving input IDs to: {args.save_input_ids}")
        save_tensor(all_eval_ids, args.save_input_ids, "input_ids")

    # Output logits
    save_logits_a = []
    save_logits_b = []

    # Metrics collection
    cumulative_metrics = []
    logits_metrics = {}

    # Run cumulative mode if requested
    if args.analysis_mode in ['cumulative', 'both']:
        # Inference
        for idx, (module_a, module_b) in enumerate(zip(model_a.modules, model_b.modules)):

            logits_layer = module_a == model_a.modules[-1]

            # Load modules
            config_a.stc.begin_deferred_load()
            module_a.load(device if not module_a.caps.get("prefer_cpu") else "cpu")
            config_a.stc.end_deferred_load()

            config_b.stc.begin_deferred_load()
            module_b.load(device if not module_b.caps.get("prefer_cpu") else "cpu")
            config_b.stc.end_deferred_load()

            # Error measures
            max_diff = 0
            rfn_error_sum = 0
            cos_error_sum = 0
            sqnr_sum = 0

            # Similarity measures
            topk_max = args.topk_max
            logprob_sum = [0, 0]
            logprob_count = [0, 0]
            kl_div_sum_ab = 0
            kl_div_sum_ba = 0
            topk_hits_sum = [[0] * topk_max, [0] * topk_max]
            topk_hits_count = [[0] * topk_max, [0] * topk_max]
            topk_agreement_sum = [0] * topk_max
            topk_agreement_count = [0] * topk_max

            for b in range(len(states_a)):

                # Advance state
                state_a = states_a[b]
                state_b = states_b[b]
                eval_ids = all_eval_ids[b]

                params_a = {}
                state_a = module_a.prepare_for_device(state_a, params_a)
                state_a = module_a.forward(state_a, params_a)

                params_b = {}
                state_b = module_b.prepare_for_device(state_b, params_b)
                state_b = module_b.forward(state_b, params_b)

                # Optionally override model A state for first layers
                if idx < args.keep_b:
                    state_a = state_b.clone()

                # Drop logits on last iteration
                if not logits_layer:
                    states_a[b] = state_a
                    states_b[b] = state_b

                # Copy logits to CPU if saving
                else:
                    if save_logits_a:
                        save_logits_a.append(state_a.cpu().split(1))
                    if save_logits_b:
                        save_logits_b.append(state_b.cpu().split(1))

                # Measure error
                if not logits_layer:
                    rows = state_a.shape[0]
                    for j in range(rows):
                        sa = state_a[j].to(float)
                        sb = state_b[j].to(float)
                        cos_error_sum += cosine_error(sa, sb)
                        sqnr_sum += sqnr(sa, sb)
                        sa -= sb
                        rfn_error_sum += (torch.linalg.norm(sa, 'fro') / torch.linalg.norm(sb, 'fro').mean()).item()
                        sa.abs_()
                        md = ((sa.max().item()) / torch.linalg.norm(sb, 'fro').mean()).item()
                        max_diff = max(max_diff, md)
                        del sa, sb

                # Perplexity, KL-div
                if logits_layer:
                    rows = state_a.shape[0]
                    for j in range(rows):
                        x = (state_a[j], state_b[j])
                        input_ids = eval_ids[j]
                        top_indices = []

                        for i in [0, 1]:
                            logits = x[i][:-1, :]
                            logprob_sum__, logprob_count__ = ppl(input_ids, logits)
                            logprob_sum[i] += logprob_sum__
                            logprob_count[i] += logprob_count__

                            _, top_index = torch.topk(logits, topk_max, dim = -1)
                            top_index = top_index.cpu().view(-1, topk_max)
                            top_indices.append(top_index)
                            targets = input_ids[1:].view(-1, 1)

                            for t in range(topk_max):
                                top_slice = top_index[:, :t + 1]
                                hits = torch.eq(targets, top_slice)
                                row_hits = hits.any(dim = 1)
                                topk_hits_sum[i][t] += row_hits.sum().item()
                                topk_hits_count[i][t] += top_slice.shape[0]

                        for t in range(topk_max):
                            top_slice_a = top_indices[0][:, :t + 1]
                            top_slice_b = top_indices[1][:, :t + 1]
                            hits = torch.eq(top_slice_a, top_slice_b)
                            row_hits = hits.all(dim = 1)
                            topk_agreement_sum[t] += row_hits.sum().item()
                            topk_agreement_count[t] += top_slice_a.shape[0]

                        epsilon = 1e-10
                        probs_a = torch.softmax(x[0].float(), dim = -1)
                        probs_b = torch.softmax(x[1].float(), dim = -1)
                        kl_div = F.kl_div(torch.log(probs_a + epsilon), probs_b, reduction = 'none')
                        kl_div_sum_ab += kl_div.sum(dim = -1).mean().item()
                        kl_div = F.kl_div(torch.log(probs_b + epsilon), probs_a, reduction = 'none')
                        kl_div_sum_ba += kl_div.sum(dim = -1).mean().item()

            # Print error
            if not logits_layer:
                rfn_error = rfn_error_sum / args.rows
                cos_error = cos_error_sum / args.rows
                sqnr_ = sqnr_sum / args.rows
                print(
                    f" -- {module_a.key:40}"
                    f"   rfn_err: {rfn_error:.6f}"
                    f"   max_diff/norm: {max_diff:.6f}"
                    f"   sqnr: {sqnr_:9.6f}"
                    f"   cos_err: {cos_error:.6f}"
                )

                # Collect metrics
                cumulative_metrics.append({
                    'layer_idx': idx,
                    'module_key': module_a.key,
                    'rfn_error': rfn_error,
                    'max_diff': max_diff,
                    'sqnr': sqnr_,
                    'cos_error': cos_error
                })

            # Save logits
            if logits_layer:
                if args.save_logits_a:
                    print(f" -- Saving model A logits to: {args.save_logits_a}")
                    save_tensor(state_a, args.save_logits_a, "logits")
                if args.save_logits_b:
                    print(f" -- Saving model B logits to: {args.save_logits_b}")
                    save_tensor(state_b, args.save_logits_b, "logits")

            # Final ppl, kld
            if logits_layer:
                perplexity = [math.exp(-logprob_sum[i] / logprob_count[i]) for i in (0, 1)]
                kl_div_ab = kl_div_sum_ab / args.rows
                kl_div_ba = kl_div_sum_ba / args.rows

                # Collect logits metrics
                logits_metrics = {
                    'perplexity_a': perplexity[0],
                    'perplexity_b': perplexity[1],
                    'kl_div_ab': kl_div_ab,
                    'kl_div_ba': kl_div_ba
                }

            # Unload modules
            module_a.unload()
            config_a.stc.close()
            free_mem()

            module_b.unload()
            config_b.stc.close()
            free_mem()

        # Perplexity for each model
        print(f" -- A perplexity: {perplexity[0]:11.8f}")
        print(f" -- B perplexity: {perplexity[1]:11.8f}")

        # Probability of the test label being in the top K tokens, for each model
        print(f" -- A label in top-K:")
        for t in range(topk_max):
            a_acc_ = topk_hits_sum[0][t] / topk_hits_count[0][t]
            print(f"      K = {t+1}: {a_acc_:6.4f}")
        print(f" -- B label in top-K:")
        for t in range(topk_max):
            a_acc_ = topk_hits_sum[1][t] / topk_hits_count[1][t]
            print(f"      K = {t+1}: {a_acc_:6.4f}")

        # Probability of exact top-K token match between models
        print(f" -- Top-K agreement, A vs B:")
        for t in range(topk_max):
            topk_agree_ = topk_agreement_sum[t] / topk_agreement_count[t]
            print(f"      K = {t+1}: {topk_agree_:6.4f}")

        # KLD, either way around
        print(f" -- KL divergence (A, B): {kl_div_ab:11.8f}")
        print(f" -- KL divergence (B, A): {kl_div_ba:11.8f}")

        # Print cumulative mode summary
        print_summary(cumulative_metrics, "CUMULATIVE", args.top_n_worst)

    # Run isolated mode if requested
    isolated_metrics = []
    if args.analysis_mode in ['isolated', 'both']:
        isolated_metrics = run_isolated_pass(
            model_a, model_b, config_a, config_b,
            all_eval_ids, device, args
        )
        print_summary(isolated_metrics, "ISOLATED", args.top_n_worst)

    # Save metrics to JSON if requested
    if args.save_layer_metrics:
        save_metrics_json(cumulative_metrics, isolated_metrics, logits_metrics, args, args.save_layer_metrics)

    # Generate plots if requested
    if args.plot_layer_metrics:
        plot_metrics(cumulative_metrics, isolated_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ma", "--model_a", type = str, help = "Model A", required = True)
    parser.add_argument("-mb", "--model_b", type = str, help = "Model B", required = True)
    parser.add_argument("-r", "--rows", type = int, help = "Number of rows", default = 100)
    parser.add_argument("-kb", "--keep_b", type = int, help = "Maintain B state for number of modules", default = 0)
    parser.add_argument("-tkm", "--topk_max", type = int, default = 5, help = "Max top-K interval to test")
    parser.add_argument("-d", "--device", type = int, help = "CUDA device index", default = 0)
    parser.add_argument("-or", "--override", type = str, help = "Model A tensor override spec (YAML)", default = None)
    parser.add_argument("-si", "--save_input_ids", type = str, help = "Save input IDs (filename)", default = None)
    parser.add_argument("-sla", "--save_logits_a", type = str, help = "Save model A logits (filename)", default = None)
    parser.add_argument("-slb", "--save_logits_b", type = str, help = "Save model B logits (filename)", default = None)
    parser.add_argument("-bsz", "--batch_size", type = int, help = "Batch size", default = 1)
    parser.add_argument("--analysis_mode", type = str, default = "cumulative",
                        choices = ["cumulative", "isolated", "both"],
                        help = "Analysis mode: cumulative (default), isolated, or both")
    parser.add_argument("--top_n_worst", type = int, default = 10,
                        help = "Show top N worst layers in summary (default: 10)")
    parser.add_argument("--save_layer_metrics", type = str, default = None,
                        help = "Save per-layer metrics to JSON file")
    parser.add_argument("--plot_layer_metrics", action = "store_true",
                        help = "Generate matplotlib plot of per-layer metrics")
    _args = parser.parse_args()
    main(_args)
