import json
from pathlib import Path
import tempfile
import unittest
from summarize_runs import summarize


class SummaryTests(unittest.TestCase):
    def make_runs(self, root):
        paths = []
        for method, score in [('fft', .5), ('lora', .4), ('dora', .45)]:
            for seed in [42, 123, 1024]:
                path = root / f'{method}_{seed}'
                path.mkdir()
                config = dict(method=method, training_seed=seed, manifest_sha256='fixed', model_commit='fixed',
                              precision='float32', versions={}, scheduler=None, max_epochs=2,
                              checkpoint_selection='validation F1', metrics='test protocol', hardware={},
                              arguments={k: 'fixed' for k in ['model', 'revision', 'device', 'batch_size', 'max_length',
                                                              'stride', 'max_answer_tokens', 'validations_per_epoch']},
                              optimizer={'lr': .001}, adapter_modules=[], trainable_parameters=20)
                metrics = {k: score for k in ['em', 'f1', 'question_macro_em', 'question_macro_f1']}
                result = dict(config=config, test=metrics,
                              resources=dict(training_seconds=1., sampled_maxima_bytes={'process_rss_bytes': 100}))
                records = [dict(example_id=str(i), group_id=str(i // 2), references=['answer'], em=score, f1=score) for i in range(6)]
                (path / 'result.json').write_text(json.dumps(result))
                (path / 'test_predictions.json').write_text(json.dumps(records))
                paths.append(path)
        return paths

    def test_paired_delta_and_sample_sd(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self.make_runs(Path(d))
            report = summarize(paths, n_bootstrap=50)
            self.assertAlmostEqual(report['paired_comparisons']['dora_minus_lora']['f1']['difference'], .05)
            self.assertAlmostEqual(report['summary']['fft']['f1']['std'], 0.)

    def test_missing_seed_and_changed_references_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self.make_runs(Path(d))
            with self.assertRaises(ValueError):
                summarize(paths[:-1], n_bootstrap=10)
            file = paths[-1] / 'test_predictions.json'
            records = json.loads(file.read_text())
            records[0]['references'] = ['different']
            file.write_text(json.dumps(records))
            with self.assertRaises(ValueError):
                summarize(paths, n_bootstrap=10)

    def test_mixed_learning_rates_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self.make_runs(Path(d))
            file = paths[-1] / 'result.json'
            result = json.loads(file.read_text())
            result['config']['optimizer']['lr'] = .002
            file.write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                summarize(paths, n_bootstrap=10)


if __name__ == '__main__':
    unittest.main()
