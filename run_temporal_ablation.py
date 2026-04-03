#!/usr/bin/env python3
"""
Temporal Frame Sampling Ablation for STEM Pipeline.
Tests 4 sampling strategies: early (25%), middle (50%), late (75%), multi-frame majority vote.
Runs VLM (Qwen2-VL-7B-Instruct) inference on all 4,000 videos × 4 strategies.
"""

import json, os, sys, time, argparse
import numpy as np
import torch
from pathlib import Path
from collections import Counter

# ============================================================
# Config
# ============================================================
CODES_DIR = Path("/home/dongwoo39/papers/paper_DIAGONAL/CODES_DIAGONAL")
OUTPUT_DIR = CODES_DIR / "outputs"
DATA_DIR = Path("streamlit_app/data")
RESULT_FILE = DATA_DIR / "temporal_ablation_results.json"

MODELS = ["storydiff", "echoshot", "vgot", "vic"]
PATTERNS = ["Relay", "Split", "Accumulation", "Convergence",
            "Sliding_Window", "Reduction", "Reverse_Relay"]

# Frame positions within each shot (as fraction)
STRATEGIES = {
    "early":  [0.25],
    "middle": [0.50],
    "late":   [0.75],
    "multi3": [0.25, 0.50, 0.75],  # majority vote over 3 frames
}

NUM_SHOTS = 3


def get_video_path(model, pattern, vid):
    """Find video file path."""
    return OUTPUT_DIR / model / pattern / f"{vid}.mp4"


def extract_frames(video_path, shot_fractions):
    """Extract frames at given fractional positions within each of 3 shots."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_shot = total_frames // NUM_SHOTS

    all_frames = []
    for shot_idx in range(NUM_SHOTS):
        shot_start = shot_idx * frames_per_shot
        shot_frames = []
        for frac in shot_fractions:
            if frac == 0.50:
                # Match main eval: shot_len // 2 (consistent with eval_vlm_fullscale.py)
                frame_idx = shot_start + frames_per_shot // 2
            else:
                frame_idx = shot_start + int(frac * (frames_per_shot - 1))
            frame_idx = min(frame_idx, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                shot_frames.append(frame)
            else:
                shot_frames.append(None)
        all_frames.append(shot_frames)
    cap.release()
    return all_frames  # shape: [3 shots][k frames per shot]


def setup_vlm(gpu_id):
    """Load Qwen2-VL model on specified GPU."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map=f"cuda:{gpu_id}",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    return model, processor


def query_vlm(model, processor, frame, entity, device):
    """Query VLM: Is entity visible? Returns 1 (Yes) or 0 (No)."""
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    if frame is None:
        return 0

    img = Image.fromarray(frame)

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": f"Is there a {entity} visible in this image? Answer only Yes or No."}
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    generated = output_ids[0][inputs.input_ids.shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True).strip().lower()
    return 1 if response.startswith("yes") else 0


def compute_metrics(S_obs, S_ideal):
    """Compute PTM and EPA from observed and ideal presence matrices."""
    S_obs = np.array(S_obs)
    S_ideal = np.array(S_ideal)

    # Delta S
    dS_obs = S_obs[1:] - S_obs[:-1]
    dS_ideal = S_ideal[1:] - S_ideal[:-1]

    # Error matrix
    E = dS_obs - dS_ideal

    # PTM: prescribed transition match
    mask = (dS_ideal != 0).astype(int)
    n_prescribed = mask.sum()
    if n_prescribed == 0:
        ptm = 1.0
    else:
        errors_on_prescribed = ((E * mask) != 0).sum()
        ptm = 1.0 - errors_on_prescribed / n_prescribed

    # EPA
    T, N = S_obs.shape
    epa = 1.0 - np.sum(S_obs != S_ideal) / (T * N)

    return float(ptm), float(epa)


def process_batch(gpu_id, entries, strategies):
    """Process a batch of entries on a single GPU."""
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading VLM...")
    model, processor = setup_vlm(gpu_id)
    print(f"[GPU {gpu_id}] Processing {len(entries)} entries...")

    results = []
    for idx, entry in enumerate(entries):
        vid = entry["vid"]
        method = entry["method"]
        pattern = entry["pattern"]
        entities = entry["entities"]
        S_ideal = entry["S_ideal"]

        video_path = get_video_path(method, pattern, vid)
        if not video_path.exists():
            print(f"[GPU {gpu_id}] WARNING: {video_path} not found, skipping")
            continue

        result_entry = {
            "method": method, "vid": vid, "pattern": pattern,
            "entities": entities, "S_ideal": S_ideal,
        }

        for strat_name, fractions in strategies.items():
            # Extract frames
            all_frames = extract_frames(video_path, fractions)

            # Query VLM for each shot × entity
            S_obs = []
            for shot_idx in range(NUM_SHOTS):
                shot_presence = []
                for entity in entities:
                    if len(fractions) == 1:
                        # Single frame
                        val = query_vlm(model, processor, all_frames[shot_idx][0], entity, device)
                    else:
                        # Multi-frame majority vote
                        votes = [query_vlm(model, processor, f, entity, device)
                                 for f in all_frames[shot_idx] if f is not None]
                        val = 1 if sum(votes) > len(votes) / 2 else 0
                    shot_presence.append(val)
                S_obs.append(shot_presence)

            ptm, epa = compute_metrics(S_obs, S_ideal)
            result_entry[f"S_obs_{strat_name}"] = S_obs
            result_entry[f"ptm_{strat_name}"] = ptm
            result_entry[f"epa_{strat_name}"] = epa

        results.append(result_entry)

        if (idx + 1) % 50 == 0:
            print(f"[GPU {gpu_id}] {idx+1}/{len(entries)} done")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--analyze_only", action="store_true")
    args = parser.parse_args()

    if args.analyze_only:
        analyze_results()
        return

    # Load original data
    with open(DATA_DIR / "vlm_fullscale_merged.json") as f:
        all_data = json.load(f)

    print(f"Total entries: {len(all_data)}")

    # Split across GPUs
    chunk_size = len(all_data) // args.num_gpus
    start = args.gpu * chunk_size
    end = start + chunk_size if args.gpu < args.num_gpus - 1 else len(all_data)
    my_entries = all_data[start:end]

    print(f"GPU {args.gpu}: processing entries {start}-{end} ({len(my_entries)} entries)")

    results = process_batch(args.gpu, my_entries, STRATEGIES)

    # Save per-GPU results
    out_path = DATA_DIR / f"temporal_ablation_gpu{args.gpu}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {out_path}")


def analyze_results():
    """Analyze and print temporal ablation results."""
    # Try to load merged results, otherwise merge per-GPU files
    if RESULT_FILE.exists():
        with open(RESULT_FILE) as f:
            data = json.load(f)
    else:
        data = []
        for gpu_id in range(4):
            path = DATA_DIR / f"temporal_ablation_gpu{gpu_id}.json"
            if path.exists():
                with open(path) as f:
                    data.extend(json.load(f))
        if data:
            with open(RESULT_FILE, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Merged {len(data)} results into {RESULT_FILE}")

    if not data:
        print("No results found!")
        return

    print(f"\n{'='*70}")
    print(f"TEMPORAL FRAME SAMPLING ABLATION ({len(data)} videos)")
    print(f"{'='*70}")

    strategies = ["early", "middle", "late", "multi3"]
    models = ["storydiff", "echoshot", "vgot", "vic"]
    model_names = {"storydiff": "SD", "echoshot": "Echo", "vgot": "VGoT", "vic": "VIC"}

    # Per-model, per-strategy PTM and EPA
    print(f"\n1. PTM BY MODEL × STRATEGY")
    print(f"   {'Model':<8}", end="")
    for s in strategies:
        print(f" {s:>8}", end="")
    print()
    print("   " + "-" * 42)

    rankings = {}
    for s in strategies:
        model_ptms = {}
        for m in models:
            entries = [e for e in data if e["method"] == m]
            ptms = [e[f"ptm_{s}"] for e in entries if f"ptm_{s}" in e]
            model_ptms[m] = np.mean(ptms) if ptms else 0
        rankings[s] = sorted(models, key=lambda m: model_ptms[m], reverse=True)

    for m in models:
        print(f"   {model_names[m]:<8}", end="")
        for s in strategies:
            entries = [e for e in data if e["method"] == m]
            ptms = [e[f"ptm_{s}"] for e in entries if f"ptm_{s}" in e]
            mean_ptm = np.mean(ptms) if ptms else 0
            print(f" {mean_ptm:>8.3f}", end="")
        print()

    # Rankings comparison
    print(f"\n2. MODEL RANKINGS BY STRATEGY")
    for s in strategies:
        rank_str = " > ".join([model_names[m] for m in rankings[s]])
        print(f"   {s:<8}: {rank_str}")

    # Rank preservation
    baseline_rank = rankings["middle"]
    print(f"\n3. RANK PRESERVATION vs MIDDLE")
    for s in strategies:
        match = rankings[s] == baseline_rank
        print(f"   {s:<8}: {'✓ PRESERVED' if match else '✗ DIFFERENT'} {rankings[s]}")

    # Spearman correlation between strategies
    from scipy.stats import spearmanr
    print(f"\n4. SPEARMAN CORRELATION BETWEEN STRATEGIES (per-video PTM)")
    for s1 in strategies:
        for s2 in strategies:
            if s1 >= s2:
                continue
            ptms1 = [e.get(f"ptm_{s1}", None) for e in data]
            ptms2 = [e.get(f"ptm_{s2}", None) for e in data]
            valid = [(a, b) for a, b in zip(ptms1, ptms2) if a is not None and b is not None]
            if valid:
                a, b = zip(*valid)
                r, p = spearmanr(a, b)
                print(f"   {s1} vs {s2}: ρ = {r:.3f} (p = {p:.2e})")

    # EPA analysis
    print(f"\n5. EPA BY MODEL × STRATEGY")
    print(f"   {'Model':<8}", end="")
    for s in strategies:
        print(f" {s:>8}", end="")
    print()
    print("   " + "-" * 42)
    for m in models:
        print(f"   {model_names[m]:<8}", end="")
        for s in strategies:
            entries = [e for e in data if e["method"] == m]
            epas = [e[f"epa_{s}"] for e in entries if f"epa_{s}" in e]
            mean_epa = np.mean(epas) if epas else 0
            print(f" {mean_epa:>8.3f}", end="")
        print()


if __name__ == "__main__":
    main()
