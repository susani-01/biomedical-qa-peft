"""Generate numbered figures only from verified matched runs; no hard-coded scores."""
import argparse
import json
from pathlib import Path
import numpy as np
from summarize_runs import summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = summarize(args.runs)
    args.out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'legend.fontsize': 11})
    colors = dict(fft='#0072B2', lora='#D55E00', dora='#009E73')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for method, values in report['summary'].items():
        for ax, metric in zip(axes, ['em', 'f1']):
            ax.errorbar(values['trainable_parameters'] / 1e6, values[metric]['mean'],
                        yerr=values[metric]['std'], fmt='o', capsize=5,
                        color=colors[method], label=method.upper())
            ax.set(xscale='log', xlabel='Trainable parameters (millions)', ylabel=metric.upper())
            ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle('Test performance: mean ± sample SD across training seeds')
    for extension in ['png', 'pdf']:
        fig.savefig(args.out / f'Figure_2_test_performance.{extension}', dpi=300)
    plt.close(fig)
    histories = {method: [] for method in colors}
    for path in args.runs:
        config = json.loads((path / 'config.json').read_text())
        histories[config['method']].append(json.loads((path / 'history.json').read_text()))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for method, runs in histories.items():
        x = np.array([entry['epoch'] for entry in runs[0]])
        if any(not np.array_equal(x, [entry['epoch'] for entry in run]) for run in runs):
            raise ValueError('Validation schedules differ across seeds; do not silently interpolate')
        for ax, metric in zip(axes, ['loss', 'em', 'f1']):
            y = np.array([[entry['validation'][metric] for entry in run] for run in runs])
            mean, sd = y.mean(axis=0), y.std(axis=0, ddof=1)
            ax.plot(x, mean, 'o-', color=colors[method], label=method.upper())
            ax.fill_between(x, mean - sd, mean + sd, color=colors[method], alpha=.15)
            ax.set(xlabel='Epoch', ylabel='Validation ' + metric.upper())
            ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle('Validation trajectory: mean ± sample SD across training seeds')
    for extension in ['png', 'pdf']:
        fig.savefig(args.out / f'Figure_3_validation_curves.{extension}', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
