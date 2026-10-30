"""One fresh FFT/LoRA/DoRA run. Run each seed in a separate process.

No legacy checkpoints are loaded. Validation selects a checkpoint; test is
evaluated once afterwards. This script intentionally does not tune on test data.
"""
import argparse
from collections import defaultdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import threading
import time

from qa_core import assert_independent, best_span, digest, make_features, score_answer


def seed_all(seed, torch, np):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Unsupported deterministic operations may still be nondeterministic on MPS.
    torch.use_deterministic_algorithms(True, warn_only=True)


def synchronize(torch, device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


class ResourceMonitor:
    """Sampled maxima, not guaranteed true allocation peaks. Never sum MPS and RSS."""
    def __init__(self, torch, device, interval=.02):
        import psutil
        self.torch, self.device, self.interval = torch, device, interval
        self.process = psutil.Process()
        self.maxima = defaultdict(int)
        self.samples, self.errors, self.seconds = 0, [], 0.

    def sample(self):
        values = {'process_rss_bytes': self.process.memory_info().rss}
        if self.device.type == 'mps':
            values.update(mps_tensor_bytes=self.torch.mps.current_allocated_memory(),
                          mps_driver_bytes=self.torch.mps.driver_allocated_memory())
        elif self.device.type == 'cuda':
            values['cuda_tensor_bytes'] = self.torch.cuda.memory_allocated(self.device)
        for name, value in values.items():
            self.maxima[name] = max(self.maxima[name], value)
        self.samples += 1

    def _poll(self):
        while not self.stop.wait(self.interval):
            try:
                self.sample()
            except Exception as error:
                self.errors.append(str(error))
                break

    def start(self):
        synchronize(self.torch, self.device)
        self.stop = threading.Event()
        self.sample()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.started = time.perf_counter()
        self.thread.start()

    def finish(self):
        synchronize(self.torch, self.device)
        self.seconds += time.perf_counter() - self.started
        self.stop.set()
        self.thread.join()
        self.sample()

    def result(self):
        return dict(training_seconds=self.seconds, sampled_maxima_bytes=dict(self.maxima),
                    sampling_interval_seconds=self.interval, samples=self.samples, errors=self.errors,
                    scope='Training batches only: includes tensor transfer, forward/backward, optimizer and monitoring; excludes tokenization, validation, checkpoint I/O, test.',
                    precision='Sampled maxima can miss short-lived peaks. RSS and MPS share unified memory; do not add them.')


def collate(features, torch):
    keys = ['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions']
    return {key: torch.tensor([f[key] for f in features], dtype=torch.long)
            for key in keys if key in features[0]}


def evaluate(model, examples, features, batch_size, device, torch, max_answer_tokens):
    model.eval()
    by_id = {item['example_id']: item for item in examples}
    predictions, loss_sum = {}, 0.
    with torch.inference_mode():
        for begin in range(0, len(features), batch_size):
            chunk = features[begin:begin + batch_size]
            batch = {k: v.to(device) for k, v in collate(chunk, torch).items()}
            output = model(**batch)
            loss_sum += float(output.loss.item()) * len(chunk)
            starts, ends = output.start_logits.cpu().tolist(), output.end_logits.cpu().tolist()
            for feature, start, end in zip(chunk, starts, ends):
                eid = feature['example_id']
                candidate = best_span(start, end, feature['offsets'], by_id[eid]['context'], max_answer_tokens)
                if candidate and (eid not in predictions or candidate['score'] > predictions[eid]['score']):
                    predictions[eid] = candidate
    records, question_scores = [], defaultdict(list)
    for item in examples:
        candidate = predictions.get(item['example_id'])
        if candidate is None:
            raise ValueError(f'No valid prediction for {item["example_id"]}')
        references = list(dict.fromkeys(a['text'] for a in item['answers']))
        scores = score_answer(candidate['text'], references)
        records.append(dict(example_id=item['example_id'], group_id=item['group_id'],
                            question_ids=item['question_ids'], prediction=candidate['text'],
                            references=references, **scores))
        for qid in item['question_ids']:
            question_scores[qid].append(scores)
    metrics = dict(loss=loss_sum / len(features), examples=len(records), questions=len(question_scores))
    for metric in ['em', 'f1']:
        metrics[metric] = sum(r[metric] for r in records) / len(records)
        metrics['question_macro_' + metric] = sum(sum(s[metric] for s in values) / len(values)
                                                 for values in question_scores.values()) / len(question_scores)
    return metrics, records


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path(__file__).parent / 'artifacts/split_manifest.json')
    p.add_argument('--method', choices=['fft', 'lora', 'dora'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--model', default='michiyasunaga/BioLinkBERT-base')
    p.add_argument('--revision', default=None, help='Prefer an immutable model commit, or a local snapshot path')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--out', type=Path, required=True, help='Must be a new directory')
    p.add_argument('--device', choices=['mps', 'cpu', 'cuda'], default='mps')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=None)
    p.add_argument('--max-length', type=int, default=384)
    p.add_argument('--stride', type=int, default=128)
    p.add_argument('--max-answer-tokens', type=int, default=30)
    p.add_argument('--validations-per-epoch', type=int, default=4)
    p.add_argument('--validation-only', action='store_true', help='For development/LR selection: do not evaluate test data')
    return p.parse_args()


def main():
    args = parse_args()
    if min(args.epochs, args.batch_size, args.validations_per_epoch, args.max_answer_tokens) < 1:
        raise ValueError('Epochs, batch size, validation frequency and span length must be positive')
    import numpy as np
    import torch
    import psutil
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    device = torch.device(args.device)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable; do not silently substitute CPU in a hardware comparison')
    manifest = json.loads(args.manifest.read_text())
    expected_hash = manifest.pop('manifest_sha256')
    if digest(manifest) != expected_hash:
        raise ValueError('Split manifest checksum mismatch')
    assert_independent(manifest['splits'])
    args.out.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed, torch, np)
    loading = dict(local_files_only=args.local_files_only)
    if args.revision:
        loading['revision'] = args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, **loading)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model, **loading)
    model.float()
    if args.method != 'fft':
        config = LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, bias='none',
                            target_modules=['query', 'key', 'value', 'dense'],
                            modules_to_save=['qa_outputs'], task_type=TaskType.QUESTION_ANS,
                            use_dora=args.method == 'dora')
        model = get_peft_model(model, config)
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    if not any('qa_outputs' in name for name, _ in trainable):
        raise RuntimeError('QA head is not trainable')
    lr = args.learning_rate or dict(fft=4e-5, lora=3e-4, dora=2e-4)[args.method]
    if lr <= 0:
        raise ValueError('Learning rate must be positive')
    config = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  method=args.method, training_seed=args.seed, split_seed=manifest['split_seed'],
                  manifest_sha256=expected_hash, model_commit=getattr(model.config, '_commit_hash', None),
                  optimizer=dict(name='AdamW', lr=lr, betas=[.9, .999], eps=1e-8, weight_decay=.01),
                  scheduler=None, precision='float32', gradient_accumulation=1, gradient_clipping=None,
                  max_epochs=args.epochs, checkpoint_selection='maximum validation pair F1; earliest on ties',
                  trainable_parameters=sum(n for _, n in trainable),
                  total_parameters=sum(p.numel() for p in model.parameters()), trainable_parameter_names=trainable,
                  adapter_modules=[name for name, module in model.named_modules() if hasattr(module, 'lora_A')],
                  versions={name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'peft', 'numpy', 'psutil']},
                  platform=platform.platform(), processor=platform.processor(), python=platform.python_version(),
                  hardware=dict(machine=platform.machine(), cpu_count=os.cpu_count(),
                                physical_memory_bytes=psutil.virtual_memory().total, torch_threads=torch.get_num_threads()),
                  mps_fallback=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
                  deterministic='torch deterministic algorithms requested with warn_only=True; not a bitwise reproducibility guarantee',
                  metrics='Normalized EM/token F1 against original references, max over aliases; pair mean primary, original-question macro of pair scores secondary. Not official BioASQ factoid MRR.')
    (args.out / 'config.json').write_text(json.dumps(config, indent=2))
    features, coverage = {}, {}
    for split, examples in manifest['splits'].items():
        if split == 'test' and args.validation_only:
            continue
        features[split], coverage[split] = make_features(examples, tokenizer, args.max_length, args.stride)
    (args.out / 'coverage.json').write_text(json.dumps(coverage, indent=2))
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                                 betas=(.9, .999), eps=1e-8, weight_decay=.01)
    seed_all(args.seed, torch, np)
    rng = random.Random(args.seed)
    monitor = ResourceMonitor(torch, device)
    best_f1, best_step, global_step, history = -1., None, 0, []
    checkpoint = args.out / 'best_state.pt'
    synchronize(torch, device)
    fit_started = time.perf_counter()
    for epoch in range(args.epochs):
        indices = list(range(len(features['train'])))
        rng.shuffle(indices)
        batches = [indices[i:i + args.batch_size] for i in range(0, len(indices), args.batch_size)]
        interval = max(1, math.ceil(len(batches) / args.validations_per_epoch))
        epoch_loss, epoch_seen = 0., 0
        model.train()
        monitor.start()
        for batch_index, indices_batch in enumerate(batches):
            batch = {k: v.to(device) for k, v in collate([features['train'][i] for i in indices_batch], torch).items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            if not torch.isfinite(output.loss).item():
                raise RuntimeError('Nonfinite training loss')
            output.loss.backward()
            optimizer.step()
            epoch_loss += output.loss.item() * len(indices_batch)
            epoch_seen += len(indices_batch)
            global_step += 1
            if (batch_index + 1) % interval == 0 or batch_index + 1 == len(batches):
                optimizer.zero_grad(set_to_none=True)
                del output, batch
                monitor.finish()
                metrics, _ = evaluate(model, manifest['splits']['validation'], features['validation'],
                                      args.batch_size, device, torch, args.max_answer_tokens)
                history.append(dict(epoch=epoch + (batch_index + 1) / len(batches), step=global_step,
                                    training_loss_running_epoch=epoch_loss / epoch_seen, validation=metrics))
                if metrics['f1'] > best_f1:
                    best_f1, best_step = metrics['f1'], global_step
                    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint)
                (args.out / 'history.json').write_text(json.dumps(history, indent=2))
                print(json.dumps(history[-1]), flush=True)
                model.train()
                if batch_index + 1 < len(batches):
                    monitor.start()
    synchronize(torch, device)
    fit_seconds = time.perf_counter() - fit_started
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    validation, _ = evaluate(model, manifest['splits']['validation'], features['validation'],
                             args.batch_size, device, torch, args.max_answer_tokens)
    test_metrics, predictions = None, None
    if not args.validation_only:
        test_metrics, predictions = evaluate(model, manifest['splits']['test'], features['test'],
                                            args.batch_size, device, torch, args.max_answer_tokens)
    result = dict(config=config, selected_step=best_step, validation=validation, test=test_metrics,
                  resources=monitor.result(), fit_including_validation_and_checkpoint_seconds=fit_seconds,
                  feature_counts={k: len(v) for k, v in features.items()})
    if predictions is not None:
        (args.out / 'test_predictions.json').write_text(json.dumps(predictions, indent=2))
    (args.out / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
