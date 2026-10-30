"""Run the agreed nine experiments sequentially on MPS, then aggregate and plot.

Fresh processes isolate model/optimizer state and allocator lifetime. Rotating
method order across seed blocks reduces a systematic order effect on resources.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = args.out.resolve()
    if shutil.disk_usage(root).free < 6 * 1024**3:
        raise RuntimeError('Need at least 6 GiB free to retain nine checkpoints and outputs')
    out.mkdir(parents=True, exist_ok=False)
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError('The agreed MPS hardware is unavailable')
    code_files = ['run_experiment.py', 'qa_core.py', 'summarize_runs.py', 'plot_runs.py']
    snapshot = out / 'source'
    snapshot.mkdir()
    for name in code_files:
        shutil.copy2(root / name, snapshot / name)
    manifest = root / 'artifacts/split_manifest.json'
    shutil.copy2(manifest, out / 'split_manifest.json')
    manifest = out / 'split_manifest.json'
    model = args.model.resolve()
    order = [(42, 'fft'), (42, 'lora'), (42, 'dora'),
             (123, 'lora'), (123, 'dora'), (123, 'fft'),
             (1024, 'dora'), (1024, 'fft'), (1024, 'lora')]
    learning_rates = dict(fft=4e-5, lora=3e-4, dora=2e-4)
    protocol = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    learning_rates=learning_rates, training_seeds=[42, 123, 1024],
                    max_epochs=2, batch_size=4, device='mps', precision='float32',
                    max_length=384, stride=128, max_answer_tokens=30,
                    validations_per_epoch=4, checkpoint_selection='maximum validation pair F1; earliest on ties',
                    order=[dict(seed=s, method=m) for s, m in order],
                    source_hashes={name: sha256(snapshot / name) for name in code_files},
                    manifest_file_sha256=sha256(manifest),
                    manifest_sha256=json.loads(manifest.read_text())['manifest_sha256'],
                    model_path=str(model), model_file_hashes={p.name: sha256(p) for p in model.iterdir() if p.is_file()},
                    chip=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                    learning_rate_basis='User confirms rates selected in prior experiments; no new search in this batch.',
                    warning='Three seeds quantify limited training variability; comparisons are conditional on this fixed corrected split.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    env = dict(os.environ, TOKENIZERS_PARALLELISM='false', MPLCONFIGDIR=str(out / 'matplotlib_cache'))
    paths, completed = [], []
    started = time.perf_counter()
    for index, (seed, method) in enumerate(order, 1):
        run_dir = out / f'{method}_seed{seed}'
        command = [sys.executable, str(snapshot / 'run_experiment.py'), '--manifest', str(manifest),
                   '--model', str(model), '--local-files-only', '--method', method, '--seed', str(seed),
                   '--learning-rate', str(learning_rates[method]), '--device', 'mps', '--epochs', '2',
                   '--batch-size', '4', '--max-length', '384', '--stride', '128', '--max-answer-tokens', '30',
                   '--validations-per-epoch', '4', '--out', str(run_dir)]
        status = dict(state='running', completed=completed, current=dict(method=method, seed=seed),
                      elapsed_seconds=time.perf_counter()-started)
        (out / 'status.json').write_text(json.dumps(status, indent=2))
        print(f'START {index}/9 {method} seed={seed} lr={learning_rates[method]}', flush=True)
        log_path = out / f'{method}_seed{seed}.log'
        with log_path.open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith('{"epoch":'):
                    entry = json.loads(line)
                    print(f'PROGRESS {index}/9 {method} seed={seed} epoch={entry["epoch"]:.2f} '
                          f'step={entry["step"]} val_F1={entry["validation"]["f1"]:.4f}', flush=True)
            code = process.wait()
        if code:
            status.update(state='failed', returncode=code, log=str(log_path))
            (out / 'status.json').write_text(json.dumps(status, indent=2))
            print(log_path.read_text()[-5000:], flush=True)
            raise RuntimeError(f'Run failed: {method} seed {seed}; prior completed runs are preserved')
        result = json.loads((run_dir / 'result.json').read_text())
        if result['resources']['errors'] or not result['resources']['samples']:
            raise RuntimeError(f'Resource sampling failed for {run_dir}')
        if result['test']['examples'] != 502 or result['config']['manifest_sha256'] != protocol['manifest_sha256']:
            raise RuntimeError('Unexpected test population or split checksum')
        completed.append(dict(method=method, seed=seed, test=result['test'],
                              training_seconds=result['resources']['training_seconds'], path=str(run_dir)))
        paths.append(run_dir)
        print(f'COMPLETE {index}/9 {method} seed={seed} EM={result["test"]["em"]:.4f} F1={result["test"]["f1"]:.4f}', flush=True)
    for script, directory in [('summarize_runs.py', 'summary'), ('plot_runs.py', 'figures')]:
        command = [sys.executable, str(snapshot / script), *map(str, paths), '--out', str(out / directory)]
        with (out / f'{directory}.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    (out / 'status.json').write_text(json.dumps(dict(state='complete', completed=completed,
                                                    elapsed_seconds=time.perf_counter()-started), indent=2))
    print('ALL NINE RUNS, SUMMARY, AND FIGURES COMPLETE: ' + str(out), flush=True)


if __name__ == '__main__':
    main()
