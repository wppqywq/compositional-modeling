#!/usr/bin/env python
"""
Run speaker policy models in parallel with progress tracking.
Saves results to CSV and generates plots.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
SEED = 42
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

OUT_DIR = os.path.join(ROOT_DIR, "results", "comm")
SOURCE_SUBDIR = "programs_for_you"

# Tower split
TOWER_ORDER = ["CL", "CPi", "PiC", "LPi", "LC", "PiL"]
TRAIN_TOWERS = set(TOWER_ORDER[:4])
TEST_TOWERS = set(TOWER_ORDER[-2:])
TRAIN_NUM_TRIALS_PER_PPT = 8
TEST_NUM_TRIALS_PER_PPT = 4

# Training protocol
NUM_GENERATIONS = 50
TEST_NUM_PPTS = 5

# Lexicon
LEXEMES = ["blah", "blab", "bloop", "bleep", "floop"]

# RSA parameters
SPEAKER_ALPHA_PROG = 2.5
SPEAKER_ALPHA_UTT = 2.5
SPEAKER_BETA_COST = 0.3
EPSILON = 0.01

# Pact-Aware parameters
PACT_GAMMA = 0.5
PACT_ETA = 1.0
REPAIR_MAX_TURNS = 2

# Drift-Biased parameters
DRIFT_EPSILON = 0.3
DRIFT_TAU = 1.0
DRIFT_MU = 0.05
DRIFT_POOL_WEIGHT_PREV_GEN = 0.6

# Models
MODEL_SPECS = [
    {"model": "learning_agent"},
    {"model": "strategic_agent"},
    {"model": "strategic_agent_pact"},
    {"model": "strategic_agent_drift"},
]


def list_available_ppt_ids(data_dir: str) -> List[int]:
    out = []
    for fn in os.listdir(data_dir):
        if fn.startswith("programs_ppt_") and fn.endswith(".json"):
            core = fn[len("programs_ppt_"):-len(".json")]
            try:
                out.append(int(core))
            except ValueError:
                continue
    return sorted(out)


def trials_path(ppt_id: int) -> str:
    return os.path.join(ROOT_DIR, "data", "model", SOURCE_SUBDIR, f"programs_ppt_{ppt_id}.json")


def load_trials(ppt_id: int) -> pd.DataFrame:
    return pd.read_json(trials_path(ppt_id))


def split_train_test_trials(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["towers"].isin(list(TRAIN_TOWERS))].copy()
    test_df = df[df["towers"].isin(list(TEST_TOWERS))].copy()
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_global_dsl(ppt_ids: List[int]) -> Tuple[List[str], List[str]]:
    from model.convention_formation.lexicon import BlockLexicon
    
    dsl_tokens: Set[str] = set()
    for ppt in ppt_ids:
        df = load_trials(ppt)
        train_df, _ = split_train_test_trials(df)
        for dsl_list in train_df["dsl"].tolist():
            for tok in list(dsl_list):
                dsl_tokens.add(str(tok))
    full_dsl = sorted(list(dsl_tokens))
    utt_lex = BlockLexicon(full_dsl, list(LEXEMES))
    all_utterances = sorted(list(utt_lex.utterances))
    return full_dsl, all_utterances


def run_single_model(args: Dict[str, Any]) -> Tuple[str, pd.DataFrame]:
    """Run a single model - this runs in a separate process."""
    model_name = args["model"]
    ppt_ids = args["ppt_ids"]
    full_dsl = args["full_dsl"]
    all_utterances = args["all_utterances"]
    random_seed = args["random_seed"]
    num_generations = args["num_generations"]
    test_num_ppts = args["test_num_ppts"]
    
    # Import inside function for multiprocessing
    from model.convention_formation.distribution import LexiconPrior
    from model.speaker_policies import (
        as_float, update_beliefs, choose_utterance_rsa,
        choose_program_rsa, compute_lexeme_mapping_entropy,
        PactMemory, choose_program_pact,
        DriftPool, choose_program_drift,
    )
    
    rng = np.random.default_rng(random_seed)
    prior0 = LexiconPrior(full_dsl, list(LEXEMES))
    arch_prior = prior0
    build_prior = prior0
    
    pact_memories: Dict[int, PactMemory] = {}
    drift_pool: Optional[DriftPool] = DriftPool() if model_name == "strategic_agent_drift" else None
    
    seen_chunks_used: Set[str] = set()
    train_rows: List[Dict[str, object]] = []
    
    start_time = time.time()
    
    for gen in range(num_generations):
        ppt_id = int(rng.choice(np.array(ppt_ids, dtype=int)))
        ppt_df = load_trials(ppt_id).copy()
        train_df, _ = split_train_test_trials(ppt_df)
        
        pact_memory = None
        if model_name == "strategic_agent_pact":
            if ppt_id not in pact_memories:
                pact_memories[ppt_id] = PactMemory()
            pact_memory = pact_memories[ppt_id]
        
        # Simulate generation
        rows = []
        prog_entropies = []
        chunks_used: Set[str] = set()
        trial_repairs = []
        trial_ratified = []
        
        for _, trial in train_df.iterrows():
            actions = [str(x) for x in list(trial["dsl"])]
            programs_with_length = {str(k): int(v) for k, v in dict(trial["programs_with_length"]).items()}
            
            # Program selection
            if model_name == "strategic_agent":
                chosen_program, ent = choose_program_rsa(
                    arch_prior, programs_with_length, all_utterances, actions,
                    SPEAKER_ALPHA_PROG, SPEAKER_ALPHA_UTT, SPEAKER_BETA_COST, EPSILON, rng
                )
                prog_entropies.append(ent)
            elif model_name == "strategic_agent_pact" and pact_memory:
                chosen_program, ent = choose_program_pact(
                    arch_prior, programs_with_length, all_utterances, actions,
                    SPEAKER_ALPHA_PROG, SPEAKER_ALPHA_UTT, SPEAKER_BETA_COST, EPSILON,
                    pact_memory, PACT_GAMMA, PACT_ETA, rng
                )
                prog_entropies.append(ent)
            elif model_name == "strategic_agent_drift" and drift_pool:
                chosen_program, ent = choose_program_drift(
                    arch_prior, programs_with_length, all_utterances, actions,
                    SPEAKER_ALPHA_PROG, SPEAKER_ALPHA_UTT, SPEAKER_BETA_COST, EPSILON,
                    drift_pool, ppt_id, DRIFT_EPSILON, DRIFT_TAU, DRIFT_MU, DRIFT_POOL_WEIGHT_PREV_GEN, rng
                )
                prog_entropies.append(ent)
            else:
                programs = list(programs_with_length.keys())
                chosen_program = str(rng.choice(programs)) if programs else ""
            
            steps = chosen_program.split(" ") if chosen_program else []
            trial_repair_count = 0
            trial_all_steps_success = True
            
            for step in steps:
                is_chunk = step.startswith("chunk")
                
                if model_name == "learning_agent":
                    utt_dist = arch_prior.marginalize(lambda L: L.dsl_to_language(step))
                    utt = str(utt_dist.sample())
                else:
                    utt = choose_utterance_rsa(arch_prior, all_utterances, step, actions, SPEAKER_ALPHA_UTT, EPSILON, rng)
                
                resp_dist = build_prior.marginalize(lambda L: L.language_to_dsl(utt))
                resp = str(resp_dist.sample())
                
                acc = 1.0 if resp == step else 0.0
                rows.append({
                    "trial": int(trial["trial_num"]), "towers": str(trial["towers"]), "actions": actions,
                    "utterance": utt, "response": resp, "intention": step, "target_program": chosen_program,
                    "acc": acc, "is_chunk": 1.0 if is_chunk else 0.0, "turn_type": "main",
                })
                
                # Repair loop
                step_repair_count = 0
                while resp != step and step_repair_count < REPAIR_MAX_TURNS:
                    step_repair_count += 1
                    trial_repair_count += 1
                    if model_name == "learning_agent":
                        utt_dist = arch_prior.marginalize(lambda L: L.dsl_to_language(step))
                        utt = str(utt_dist.sample())
                    else:
                        utt = choose_utterance_rsa(arch_prior, all_utterances, step, actions, SPEAKER_ALPHA_UTT, EPSILON, rng)
                    resp_dist = build_prior.marginalize(lambda L: L.language_to_dsl(utt))
                    resp = str(resp_dist.sample())
                    acc = 1.0 if resp == step else 0.0
                    rows.append({
                        "trial": int(trial["trial_num"]), "towers": str(trial["towers"]), "actions": actions,
                        "utterance": utt, "response": resp, "intention": step, "target_program": chosen_program,
                        "acc": acc, "is_chunk": 1.0 if is_chunk else 0.0, "turn_type": "repair",
                    })
                
                if resp != step:
                    trial_all_steps_success = False
                if is_chunk:
                    chunks_used.add(step)
            
            trial_repairs.append(trial_repair_count)
            ratified = trial_repair_count == 0 and trial_all_steps_success
            trial_ratified.append(ratified)
            
            if pact_memory and model_name == "strategic_agent_pact":
                pact_memory.record_proposal(chosen_program)
                if ratified:
                    pact_memory.record_ratify(chosen_program)
                else:
                    pact_memory.record_repair(chosen_program)
            
            if drift_pool and model_name == "strategic_agent_drift":
                drift_pool.record_key_prev_gen(chosen_program)
                drift_pool.record_key_within_ppt(ppt_id, chosen_program)
        
        gen_df = pd.DataFrame(rows)
        main_rows = gen_df[gen_df["turn_type"] == "main"] if len(gen_df) > 0 else gen_df
        
        acc_comm = float(main_rows["acc"].mean()) if len(main_rows) > 0 else 0.0
        msg_len = float(main_rows.groupby("trial")["target_program"].apply(lambda s: len(s.iloc[0].split(" "))).mean()) if len(main_rows) > 0 else 0.0
        frag_rate = float(main_rows["is_chunk"].mean()) if len(main_rows) > 0 else 0.0
        program_choice_entropy = float(np.mean(prog_entropies)) if prog_entropies else 0.0
        repair_rate = sum(1 for r in trial_repairs if r > 0) / max(1, len(trial_repairs))
        
        chunks_used_now = chunks_used
        reuse_chunk_rate = len(chunks_used_now & seen_chunks_used) / max(1, len(chunks_used_now)) if chunks_used_now else 0.0
        seen_chunks_used |= chunks_used_now
        
        arch_ent = compute_lexeme_mapping_entropy(arch_prior, all_utterances, full_dsl, EPSILON)
        build_ent = compute_lexeme_mapping_entropy(build_prior, all_utterances, full_dsl, EPSILON)
        lexeme_mapping_entropy = (arch_ent + build_ent) / 2.0
        
        train_rows.append({
            "split": "train", "generation": gen, "ppt_id": ppt_id, "model": model_name,
            "acc_comm": acc_comm, "msg_len": msg_len, "frag_rate": frag_rate,
            "program_choice_entropy": program_choice_entropy, "lexeme_mapping_entropy": lexeme_mapping_entropy,
            "reuse_chunk_rate": reuse_chunk_rate, "repair_rate": repair_rate,
        })
        
        # Update beliefs
        main_rows_df = pd.DataFrame(main_rows)
        arch_prior = update_beliefs("architect", arch_prior, main_rows_df, all_utterances, EPSILON)
        build_prior = update_beliefs("builder", build_prior, main_rows_df, all_utterances, EPSILON)
        
        # Progress every 10 generations
        if (gen + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  [{model_name}] gen {gen+1}/{num_generations} ({elapsed:.1f}s)")
    
    # Test evaluation
    test_rng = np.random.default_rng(random_seed + 123)
    test_ppts = [int(x) for x in test_rng.choice(np.array(ppt_ids, dtype=int), size=test_num_ppts, replace=False)]
    test_rows: List[Dict[str, object]] = []
    
    for ppt in test_ppts:
        ppt_df = load_trials(ppt).copy()
        _, test_df = split_train_test_trials(ppt_df)
        
        test_pact_memory = pact_memories.get(ppt, PactMemory()) if model_name == "strategic_agent_pact" else None
        
        rows = []
        prog_entropies = []
        trial_repairs = []
        
        for _, trial in test_df.iterrows():
            actions = [str(x) for x in list(trial["dsl"])]
            programs_with_length = {str(k): int(v) for k, v in dict(trial["programs_with_length"]).items()}
            
            if model_name == "strategic_agent":
                chosen_program, ent = choose_program_rsa(arch_prior, programs_with_length, all_utterances, actions, SPEAKER_ALPHA_PROG, SPEAKER_ALPHA_UTT, SPEAKER_BETA_COST, EPSILON, test_rng)
                prog_entropies.append(ent)
            elif model_name == "strategic_agent_pact" and test_pact_memory:
                chosen_program, ent = choose_program_pact(arch_prior, programs_with_length, all_utterances, actions, SPEAKER_ALPHA_PROG, SPEAKER_ALPHA_UTT, SPEAKER_BETA_COST, EPSILON, test_pact_memory, PACT_GAMMA, PACT_ETA, test_rng)
                prog_entropies.append(ent)
            elif model_name == "strategic_agent_drift" and drift_pool:
                chosen_program, ent = choose_program_drift(arch_prior, programs_with_length, all_utterances, actions, SPEAKER_ALPHA_PROG, SPEAKER_ALPHA_UTT, SPEAKER_BETA_COST, EPSILON, drift_pool, ppt, DRIFT_EPSILON, DRIFT_TAU, DRIFT_MU, DRIFT_POOL_WEIGHT_PREV_GEN, test_rng)
                prog_entropies.append(ent)
            else:
                programs = list(programs_with_length.keys())
                chosen_program = str(test_rng.choice(programs)) if programs else ""
            
            steps = chosen_program.split(" ") if chosen_program else []
            trial_repair_count = 0
            
            for step in steps:
                is_chunk = step.startswith("chunk")
                if model_name == "learning_agent":
                    utt_dist = arch_prior.marginalize(lambda L: L.dsl_to_language(step))
                    utt = str(utt_dist.sample())
                else:
                    utt = choose_utterance_rsa(arch_prior, all_utterances, step, actions, SPEAKER_ALPHA_UTT, EPSILON, test_rng)
                resp_dist = build_prior.marginalize(lambda L: L.language_to_dsl(utt))
                resp = str(resp_dist.sample())
                acc = 1.0 if resp == step else 0.0
                rows.append({"trial": int(trial["trial_num"]), "acc": acc, "is_chunk": 1.0 if is_chunk else 0.0, "turn_type": "main", "target_program": chosen_program})
                
                step_repair_count = 0
                while resp != step and step_repair_count < REPAIR_MAX_TURNS:
                    step_repair_count += 1
                    trial_repair_count += 1
                    if model_name == "learning_agent":
                        utt_dist = arch_prior.marginalize(lambda L: L.dsl_to_language(step))
                        utt = str(utt_dist.sample())
                    else:
                        utt = choose_utterance_rsa(arch_prior, all_utterances, step, actions, SPEAKER_ALPHA_UTT, EPSILON, test_rng)
                    resp_dist = build_prior.marginalize(lambda L: L.language_to_dsl(utt))
                    resp = str(resp_dist.sample())
            
            trial_repairs.append(trial_repair_count)
        
        test_gen_df = pd.DataFrame(rows)
        main_rows = test_gen_df[test_gen_df["turn_type"] == "main"] if len(test_gen_df) > 0 else test_gen_df
        acc_comm = float(main_rows["acc"].mean()) if len(main_rows) > 0 else 0.0
        msg_len = float(main_rows.groupby("trial")["target_program"].apply(lambda s: len(s.iloc[0].split(" "))).mean()) if len(main_rows) > 0 else 0.0
        frag_rate = float(main_rows["is_chunk"].mean()) if len(main_rows) > 0 else 0.0
        repair_rate = sum(1 for r in trial_repairs if r > 0) / max(1, len(trial_repairs))
        
        test_rows.append({
            "split": "test", "generation": num_generations, "ppt_id": ppt, "model": model_name,
            "acc_comm": acc_comm, "msg_len": msg_len, "frag_rate": frag_rate,
            "program_choice_entropy": float(np.mean(prog_entropies)) if prog_entropies else 0.0,
            "lexeme_mapping_entropy": lexeme_mapping_entropy, "reuse_chunk_rate": 0.0, "repair_rate": repair_rate,
        })
    
    out_df = pd.concat([pd.DataFrame(train_rows), pd.DataFrame(test_rows)], ignore_index=True)
    elapsed = time.time() - start_time
    print(f"[DONE] {model_name} finished in {elapsed:.1f}s")
    return model_name, out_df


def main():
    print("=" * 60)
    print("Running Speaker Policy Models")
    print("=" * 60)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    data_dir = os.path.join(ROOT_DIR, "data", "model", SOURCE_SUBDIR)
    ppt_ids = list_available_ppt_ids(data_dir)
    
    if not ppt_ids:
        raise ValueError(f"No participant files found under {data_dir}")
    
    print(f"Found {len(ppt_ids)} participants")
    print(f"Building global DSL...")
    full_dsl, all_utterances = build_global_dsl(ppt_ids)
    print(f"DSL size: {len(full_dsl)}, Utterances: {len(all_utterances)}")
    
    # Prepare jobs
    jobs = []
    for idx, spec in enumerate(MODEL_SPECS):
        jobs.append({
            "model": spec["model"],
            "ppt_ids": ppt_ids,
            "full_dsl": full_dsl,
            "all_utterances": all_utterances,
            "random_seed": SEED + idx,
            "num_generations": NUM_GENERATIONS,
            "test_num_ppts": TEST_NUM_PPTS,
        })
    
    print(f"\nRunning {len(jobs)} models in parallel...")
    print("-" * 60)
    
    start_total = time.time()
    results = []
    
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=min(4, len(jobs))) as executor:
        futures = {executor.submit(run_single_model, job): job["model"] for job in jobs}
        for future in as_completed(futures):
            model_name, df = future.result()
            results.append(df)
    
    # Combine results
    comm_df = pd.concat(results, ignore_index=True)
    
    total_time = time.time() - start_total
    print("-" * 60)
    print(f"Total time: {total_time:.1f}s")
    
    # Save to CSV
    csv_path = os.path.join(OUT_DIR, "comm_by_gen.csv")
    comm_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAIN SUMMARY")
    print("=" * 60)
    train_sub = comm_df[comm_df["split"] == "train"]
    summary = train_sub.groupby("model")[["acc_comm", "msg_len", "frag_rate", "repair_rate"]].agg(["mean", "std"])
    print(summary.round(3))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    test_sub = comm_df[comm_df["split"] == "test"]
    test_summary = test_sub.groupby("model")[["acc_comm", "msg_len", "frag_rate", "repair_rate"]].agg(["mean", "std"])
    print(test_summary.round(3))
    
    # Generate plots
    print("\nGenerating plots...")
    final_models = ["learning_agent", "strategic_agent", "strategic_agent_pact", "strategic_agent_drift"]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.reshape(2, 3)
    
    for j, metric in enumerate(["acc_comm", "msg_len", "frag_rate"]):
        ax = axes[0, j]
        for mode in final_models:
            sub = comm_df[(comm_df["split"] == "train") & (comm_df["model"] == mode)]
            if len(sub) > 0:
                ax.plot(sub["generation"], sub[metric].rolling(window=4).mean(), label=mode, alpha=0.9)
        ax.set_title(f"{metric} (smoothed)")
        ax.grid(alpha=0.3)
        ax.set_xlabel("Generation")
    
    for j, metric in enumerate(["program_choice_entropy", "lexeme_mapping_entropy", "repair_rate"]):
        ax = axes[1, j]
        for mode in final_models:
            sub = comm_df[(comm_df["split"] == "train") & (comm_df["model"] == mode)]
            if len(sub) > 0:
                ax.plot(sub["generation"], sub[metric], label=mode, alpha=0.9)
        ax.set_title(metric)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Generation")
    
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    
    plot_path = os.path.join(OUT_DIR, "speaker_policies_comparison.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()


