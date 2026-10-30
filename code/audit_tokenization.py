"""Check real tokenizer coverage and read the exact logs used by the paper figure."""
import argparse
import json
from pathlib import Path
from qa_core import load_rows, make_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--source', required=True)
    args = p.parse_args()
    from transformers import AutoTokenizer
    from sklearn.model_selection import train_test_split
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    root = Path(__file__).resolve().parent
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = load_rows(args.source)
    _, test = train_test_split(rows, test_size=.2, random_state=42)
    counts = dict(test_examples=len(test), missing_start=0, missing_end=0,
                  both_missing=0, empty_decoded_reference=0)
    for row in test:
        encoded = tokenizer(row['question'], row['context'], max_length=384,
                            padding='max_length', truncation='only_second', return_offsets_mapping=True)
        start = end = 0
        for i, (offset, sequence) in enumerate(zip(encoded['offset_mapping'], encoded.sequence_ids())):
            if sequence != 1:
                continue
            if offset[0] <= row['answer_start'] < offset[1]:
                start = i
            if offset[0] < row['answer_start'] + len(row['answer_text']) <= offset[1]:
                end = i
                break
        counts['missing_start'] += start == 0
        counts['missing_end'] += end == 0
        counts['both_missing'] += start == end == 0
        counts['empty_decoded_reference'] += not tokenizer.decode(encoded['input_ids'][start:end+1], skip_special_tokens=True).strip()
    print('Legacy tokenization:', counts, flush=True)
    manifest = json.loads((root / 'artifacts/split_manifest.json').read_text())
    coverage = {}
    for split, items in manifest['splits'].items():
        features, covered = make_features(items, tokenizer)
        coverage[split] = dict(examples=len(items), windows=len(features),
                               references=sum(x['references'] for x in covered),
                               covered_references=sum(x['covered_references'] for x in covered),
                               examples_whose_shortest_answer_exceeds_30_tokens=sum(x['shortest_answer_tokens'] > 30 for x in covered))
        print(split, coverage[split], flush=True)
    logs = {}
    for method, relative in {'fft': 'link_model/version_6', 'lora': 'lora_seeds/version_4', 'dora': 'dora_seed/version_5'}.items():
        ea = EventAccumulator(str(root.parent / 'lightning_logs' / relative))
        ea.Reload()
        logs[method] = {tag: [dict(step=e.step, value=e.value) for e in ea.Scalars(tag)]
                        for tag in ['val_loss', 'val_exact_match']}
    report = dict(legacy_test=counts, corrected_window_coverage=coverage, figure_logs=logs)
    (root / 'artifacts/tokenization_and_curves.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(logs, indent=2))


if __name__ == '__main__':
    main()
