"""Reproduce the manuscript audit without torch, pandas, or checkpoint loading."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from qa_core import load_rows, merge_examples, split_examples, digest


def audit(source, project, source_url=None):
    rows = load_rows(source)
    # sklearn ShuffleSplit uses RandomState.permutation and ceil(test_size*n).
    perm = np.random.RandomState(42).permutation(len(rows))
    ntest = int(np.ceil(.2 * len(rows)))
    test, trainval = perm[:ntest], perm[ntest:]
    perm2 = np.random.RandomState(42).permutation(len(trainval))
    nval = int(np.ceil(.1 / .8 * len(trainval)))
    val, train = trainval[perm2[:nval]], trainval[perm2[nval:]]
    overlap = {}
    for field in ['question_id', 'question', 'context']:
        train_values = {rows[i][field] for i in train}
        overlap[field] = {
            'unique_train_test_overlap': len(train_values & {rows[i][field] for i in test}),
            'test_rows_seen_in_train': sum(rows[i][field] in train_values for i in test),
            'validation_rows_seen_in_train': sum(rows[i][field] in train_values for i in val),
        }
    saved, duplicates = [], defaultdict(list)
    for folder in ['FFT_seed_results', 'lora_seed_results', 'dora_seed_results']:
        for path in sorted((project / folder).glob('*.json')):
            result = json.loads(path.read_text())
            predictions = result['per_sample_predictions']
            empty = [p for p in predictions if not p['ground_truth'].strip()]
            fingerprint = digest(predictions)
            duplicates[(folder, fingerprint)].append(path.name)
            saved.append(dict(file=str(path.relative_to(project)), model_name=result['model_name'],
                              em=result['exact_match'], f1=result['f1_score'], n=result['num_samples'],
                              empty_references=len(empty), empty_reference_em_successes=sum(p['is_correct'] for p in empty),
                              prediction_sha256=fingerprint,
                              reference_sha256=digest([p['ground_truth'] for p in predictions])))
    examples = merge_examples(rows)
    splits = split_examples(examples)
    split_summary = {name: dict(examples=len(items),
                               original_questions=len({q for item in items for q in item['question_ids']}),
                               components=len({item['group_id'] for item in items}))
                     for name, items in splits.items()}
    hparams = {str(p.relative_to(project)): p.read_text() for p in sorted((project / 'lightning_logs').glob('*/version_*/hparams.yaml'))}
    report = dict(source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_url=source_url,
                  source_attribution='User-provided download origin; not independently byte-verified against the remote release.' if source_url else None,
                  expanded_rows=len(rows), original_question_ids=len({r['question_id'] for r in rows}),
                  distinct_question_texts=len({r['question'] for r in rows}),
                  distinct_contexts=len({r['context'] for r in rows}),
                  distinct_question_context_pairs=len(examples),
                  legacy_split_sizes=dict(train=len(train), validation=len(val), test=len(test)),
                  legacy_split_overlap=overlap, saved_evaluations=saved,
                  identical_prediction_files=[names for names in duplicates.values() if len(names) > 1],
                  recorded_training_hyperparameters=hparams,
                  proposed_split=split_summary,
                  limitations=[('Remote release number and upstream preprocessing history remain unverified.' if source_url else 'Kaggle URL/version not established from the local file.'),
                               'Filename seed labels do not prove independent training seeds.',
                               'Exact/normalized text overlap checks do not detect every near duplicate or shared document.'])
    return report, splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-url', help='User-provided download origin; recorded separately from local content verification')
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'artifacts')
    args = parser.parse_args()
    report, splits = audit(args.source, args.project, args.source_url)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'audit.json').write_text(json.dumps(report, indent=2))
    manifest = dict(schema_version=1, split_seed=42, source_sha256=report['source_sha256'],
                    policy='Connected components of original ID, normalized question, normalized context; merge exact question/context pairs; 70/10/20 components',
                    splits=splits)
    manifest['manifest_sha256'] = digest(manifest)
    (args.out / 'split_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k not in ['saved_evaluations', 'recorded_training_hyperparameters']}, indent=2))


if __name__ == '__main__':
    main()
