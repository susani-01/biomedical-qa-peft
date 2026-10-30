"""Aggregate matched, separately trained runs; bootstrap seeds and independent groups."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from qa_core import digest


def summarize(paths, n_bootstrap=5000):
    if n_bootstrap < 1:
        raise ValueError('Bootstrap replicate count must be positive')
    runs = {method: {} for method in ['fft', 'lora', 'dora']}
    common_fingerprint = None
    method_fingerprints = {}
    for path in paths:
        result = json.loads((path / 'result.json').read_text())
        config = result['config']
        if result['test'] is None:
            raise ValueError('Validation-only development runs cannot enter the final test table')
        method, seed = config['method'], config['training_seed']
        if seed in runs[method]:
            raise ValueError(f'Duplicate {method} seed {seed}')
        # Compare conditions across methods. Learning rate and adapters may differ.
        common = {k: config[k] for k in ['manifest_sha256', 'model_commit', 'precision', 'versions',
                                         'scheduler', 'max_epochs', 'checkpoint_selection', 'metrics', 'hardware']}
        common['arguments'] = {k: config['arguments'][k] for k in ['model', 'revision', 'device', 'batch_size',
                                                                'max_length', 'stride', 'max_answer_tokens', 'validations_per_epoch']}
        fingerprint = digest(common)
        if common_fingerprint is not None and fingerprint != common_fingerprint:
            raise ValueError('Runs have incompatible data, model, runtime, or evaluation settings')
        common_fingerprint = fingerprint
        method_fingerprint = digest({k: config[k] for k in ['optimizer', 'adapter_modules', 'trainable_parameters']})
        if method in method_fingerprints and method_fingerprints[method] != method_fingerprint:
            raise ValueError(f'Hyperparameters differ across {method} seeds')
        method_fingerprints[method] = method_fingerprint
        predictions = json.loads((path / 'test_predictions.json').read_text())
        ids = [p['example_id'] for p in predictions]
        if len(ids) != len(set(ids)):
            raise ValueError('Repeated example IDs in a run')
        for metric in ['em', 'f1']:
            if not np.isclose(np.mean([p[metric] for p in predictions]), result['test'][metric], atol=1e-10, rtol=0):
                raise ValueError('Saved summary disagrees with per-example scores')
        runs[method][seed] = (result, {p['example_id']: p for p in predictions})
    seeds = sorted(runs['fft'])
    if len(seeds) < 3 or any(sorted(method_runs) != seeds for method_runs in runs.values()):
        raise ValueError('Need the same >=3 distinct training seeds for all three methods')
    reference = runs['fft'][seeds[0]][1]
    ids = sorted(reference)
    for method_runs in runs.values():
        for result, predictions in method_runs.values():
            if sorted(predictions) != ids or any(
                predictions[eid]['group_id'] != reference[eid]['group_id'] or
                predictions[eid]['references'] != reference[eid]['references'] for eid in ids):
                raise ValueError('Paired comparison requires identical IDs, reference answers and groups')
    summary = {}
    for method, method_runs in runs.items():
        summary[method] = {}
        for metric in ['em', 'f1', 'question_macro_em', 'question_macro_f1']:
            values = [method_runs[seed][0]['test'][metric] for seed in seeds]
            summary[method][metric] = dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)), values=values)
        times = [method_runs[seed][0]['resources']['training_seconds'] for seed in seeds]
        summary[method]['training_seconds'] = dict(mean=float(np.mean(times)), std=float(np.std(times, ddof=1)), values=times)
        summary[method]['trainable_parameters'] = method_runs[seeds[0]][0]['config']['trainable_parameters']
        memory_keys = set.intersection(*(set(method_runs[s][0]['resources']['sampled_maxima_bytes']) for s in seeds))
        summary[method]['sampled_memory_maxima_bytes'] = {}
        for key in sorted(memory_keys):
            values = [method_runs[s][0]['resources']['sampled_maxima_bytes'][key] for s in seeds]
            summary[method]['sampled_memory_maxima_bytes'][key] = dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)), values=values)
    # Paired hierarchical bootstrap: resample training seeds and connected groups,
    # using the same draws for both methods. Keep all examples of a drawn group.
    groups = sorted({reference[eid]['group_id'] for eid in ids})
    group_indices = {g: [i for i, eid in enumerate(ids) if reference[eid]['group_id'] == g] for g in groups}
    sizes = np.array([len(group_indices[g]) for g in groups])
    comparisons = {}
    for a, b in [('dora', 'lora'), ('lora', 'fft'), ('dora', 'fft')]:
        comparisons[a + '_minus_' + b] = {}
        for metric in ['em', 'f1']:
            differences = np.array([[runs[a][seed][1][eid][metric] - runs[b][seed][1][eid][metric]
                                     for eid in ids] for seed in seeds])
            group_sums = np.stack([differences[:, group_indices[g]].sum(axis=1) for g in groups], axis=1)
            rng = np.random.default_rng(2026)
            estimates = []
            for _ in range(n_bootstrap):
                si = rng.integers(len(seeds), size=len(seeds))
                gi = rng.integers(len(groups), size=len(groups))
                estimates.append(group_sums[si][:, gi].sum() / (len(si) * sizes[gi].sum()))
            comparisons[a + '_minus_' + b][metric] = dict(difference=float(differences.mean()),
                                                         descriptive_95_percent_interval=np.quantile(estimates, [.025, .975]).tolist())
    return dict(seeds=seeds, summary=summary, paired_comparisons=comparisons,
                bootstrap_replicates=n_bootstrap,
                caution='Intervals are unadjusted descriptive paired hierarchical bootstrap intervals, not equivalence tests or proof of superiority. Three seeds give limited information about training variability.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    report = summarize(args.runs)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2))
    with (args.out / 'performance.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'seeds', 'EM_mean', 'EM_sample_std', 'F1_mean', 'F1_sample_std', 'trainable_parameters', 'training_seconds_mean'])
        for method, values in report['summary'].items():
            writer.writerow([method, len(report['seeds']), values['em']['mean'], values['em']['std'],
                             values['f1']['mean'], values['f1']['std'], values['trainable_parameters'], values['training_seconds']['mean']])
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
