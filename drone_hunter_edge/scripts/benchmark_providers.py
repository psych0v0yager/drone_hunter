#!/usr/bin/env python3
"""Benchmark ONNX execution providers and model precisions.

Runs ablation study across:
- Precisions: FP32, FP16, INT8
- Providers: CPU, XNNPACK, NNAPI

Usage:
    python benchmark_providers.py [--iterations 100] [--warmup 10]
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed")
    print("Install with: pip install onnxruntime")
    sys.exit(1)


# Model configurations
MODELS = {
    "FP32": "models/nanodet_drone_opset21.onnx",
    "FP16": "models/nanodet_drone_opset21_fp16.onnx",
    "INT8": "models/nanodet_drone_opset21_int8.onnx",
}

# Provider configurations
PROVIDERS = {
    "CPU": ["CPUExecutionProvider"],
    "XNNPACK": ["XnnpackExecutionProvider", "CPUExecutionProvider"],
    "NNAPI": ["NnapiExecutionProvider", "CPUExecutionProvider"],
}


def get_available_providers() -> List[str]:
    """Get list of available ONNX Runtime providers."""
    return ort.get_available_providers()


def check_provider_available(provider: str) -> bool:
    """Check if a provider is available."""
    available = get_available_providers()
    if provider == "CPU":
        return "CPUExecutionProvider" in available
    elif provider == "XNNPACK":
        return "XnnpackExecutionProvider" in available
    elif provider == "NNAPI":
        return "NnapiExecutionProvider" in available
    return False


def load_model(
    model_path: str,
    providers: List[str],
) -> Tuple[Optional[ort.InferenceSession], str, np.dtype]:
    """Load ONNX model with specified providers.

    Returns:
        Tuple of (session, active_provider, input_dtype)
    """
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )

        active = session.get_providers()[0]

        # Detect input dtype
        input_info = session.get_inputs()[0]
        input_dtype = np.float16 if "float16" in input_info.type else np.float32

        return session, active, input_dtype

    except Exception as e:
        return None, str(e), np.float32


def benchmark_inference(
    session: ort.InferenceSession,
    input_dtype: np.dtype,
    iterations: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """Benchmark inference speed.

    Returns:
        Dict with timing statistics (in milliseconds).
    """
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Create dummy input (320x320 image, NCHW format)
    dummy_input = np.random.rand(1, 3, 320, 320).astype(input_dtype)

    # Warmup runs
    for _ in range(warmup):
        session.run([output_name], {input_name: dummy_input})

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run([output_name], {input_name: dummy_input})
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
        "fps": float(1000.0 / np.mean(times)),
    }


def print_header():
    """Print benchmark header."""
    print("=" * 80)
    print("ONNX Runtime Provider & Precision Benchmark")
    print("=" * 80)
    print(f"Device: Android/Termux" if "TERMUX_VERSION" in __import__("os").environ else "Desktop")
    print(f"Available providers: {', '.join(get_available_providers())}")
    print("=" * 80)
    print()


def print_results_table(results: Dict[str, Dict[str, Dict]]):
    """Print results as a formatted table."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Header
    print(f"{'Precision':<10} {'Provider':<12} {'Status':<12} {'Mean (ms)':<12} {'Std (ms)':<10} {'FPS':<10}")
    print("-" * 80)

    for precision, providers in results.items():
        for provider, data in providers.items():
            if data.get("error"):
                print(f"{precision:<10} {provider:<12} {'FAILED':<12} {'-':<12} {'-':<10} {'-':<10}")
                print(f"           Error: {data['error'][:50]}")
            else:
                stats = data["stats"]
                print(f"{precision:<10} {provider:<12} {data['active']:<12} "
                      f"{stats['mean_ms']:<12.2f} {stats['std_ms']:<10.2f} {stats['fps']:<10.1f}")

    print("-" * 80)

    # Find best combination
    best_fps = 0
    best_combo = None
    for precision, providers in results.items():
        for provider, data in providers.items():
            if not data.get("error") and data["stats"]["fps"] > best_fps:
                best_fps = data["stats"]["fps"]
                best_combo = (precision, provider)

    if best_combo:
        print(f"\nBest: {best_combo[0]} + {best_combo[1]} @ {best_fps:.1f} FPS")
    print()


def run_benchmark(iterations: int = 100, warmup: int = 10, skip_nnapi: bool = False):
    """Run full benchmark suite."""
    print_header()

    # Filter providers if skipping NNAPI
    providers_to_test = {k: v for k, v in PROVIDERS.items() if not (skip_nnapi and k == "NNAPI")}
    if skip_nnapi:
        print("Skipping NNAPI (--skip-nnapi)\n")

    # Find model directory
    script_dir = Path(__file__).parent.parent
    model_dir = script_dir.parent

    results = {}

    for precision, model_rel_path in MODELS.items():
        model_path = model_dir / model_rel_path

        if not model_path.exists():
            print(f"[SKIP] {precision}: Model not found at {model_path}")
            results[precision] = {p: {"error": "Model not found"} for p in providers_to_test}
            continue

        print(f"\n[TEST] {precision}: {model_path.name}")
        results[precision] = {}

        for provider_name, provider_list in providers_to_test.items():
            print(f"  - {provider_name}...", end=" ", flush=True)

            # Check if provider is available
            if not check_provider_available(provider_name):
                print("NOT AVAILABLE")
                results[precision][provider_name] = {
                    "error": f"{provider_name} not available",
                }
                continue

            # Load model
            session, active, input_dtype = load_model(str(model_path), provider_list)

            if session is None:
                print(f"LOAD FAILED: {active}")
                results[precision][provider_name] = {
                    "error": active,
                }
                continue

            # Check if we got the provider we wanted
            expected_provider = provider_list[0]
            if active != expected_provider:
                print(f"FALLBACK ({active})")
                # Still benchmark but note the fallback

            # Run benchmark
            try:
                stats = benchmark_inference(
                    session,
                    input_dtype,
                    iterations=iterations,
                    warmup=warmup,
                )
                print(f"{stats['mean_ms']:.2f}ms ({stats['fps']:.1f} FPS)")
                results[precision][provider_name] = {
                    "active": active.replace("ExecutionProvider", ""),
                    "stats": stats,
                }
            except Exception as e:
                print(f"BENCHMARK FAILED: {e}")
                results[precision][provider_name] = {
                    "error": str(e),
                }

    print_results_table(results)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX providers and precisions"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=100,
        help="Number of inference iterations (default: 100)"
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--skip-nnapi",
        action="store_true",
        help="Skip NNAPI provider (slow on some devices)"
    )

    args = parser.parse_args()

    run_benchmark(
        iterations=args.iterations,
        warmup=args.warmup,
        skip_nnapi=args.skip_nnapi,
    )


if __name__ == "__main__":
    main()
