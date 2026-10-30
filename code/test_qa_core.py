import unittest
from qa_core import best_span, score_answer, split_examples, assert_independent, merge_examples, digest, make_features


class CoreTests(unittest.TestCase):
    def test_original_references_and_multiple_answers(self):
        self.assertEqual(score_answer('Bazex syndrome', ['another alias', 'Bazex syndrome']), {'em': 1., 'f1': 1.})
        self.assertEqual(score_answer('', ['Bazex syndrome']), {'em': 0., 'f1': 0.})
        with self.assertRaises(ValueError):
            score_answer('', [''])

    def test_span_masks_question_and_rejects_reversed_spans(self):
        # The question/CLS positions have the largest logits but cannot be decoded.
        found = best_span([100, 1, 9, 0], [100, 8, 2, 0],
                          [None, (0, 5), (6, 10), None], 'alpha beta')
        self.assertEqual(found['text'], 'beta')
        self.assertGreater(found['char_end'], found['char_start'])

    def test_answer_length_limit(self):
        found = best_span([10, 0], [0, 9], [(0, 5), (6, 10)], 'alpha beta', 1)
        self.assertEqual(found['text'], 'alpha')

    def test_connected_questions_and_contexts_stay_together(self):
        items = [dict(example_id=digest(i), question_ids=[str(i)], question=f'q{i}', context=f'c{i}') for i in range(20)]
        items[1]['question_ids'] = ['0']
        items[2]['context'] = 'c1'
        items[3]['question'] = ' Q2 '
        splits = split_examples(items)
        owners = {x['example_id']: name for name, xs in splits.items() for x in xs}
        self.assertEqual(len({owners[items[i]['example_id']] for i in range(4)}), 1)
        self.assertEqual(splits, split_examples(items))
        assert_independent(splits)

    def test_merge_preserves_answer_variants(self):
        rows = [dict(question='q', context='alpha beta', question_id='id', expanded_id=str(i),
                     answer_start=s, answer_text=t) for i, (s, t) in enumerate([(0, 'alpha'), (6, 'beta'), (0, 'alpha')])]
        items = merge_examples(rows)
        self.assertEqual(len(items), 1)
        self.assertEqual(len(items[0]['answers']), 2)
        self.assertEqual(len(items[0]['source_ids']), 3)

    def test_window_labels_are_both_cls_if_answer_is_partial(self):
        class Encoding(dict):
            def sequence_ids(self, window):
                return [None, 0, None, 1, 1, None]
        class Tokenizer:
            cls_token_id = 101
            def __call__(self, *args, **kwargs):
                return Encoding(input_ids=[[101, 1, 102, 2, 3, 102]] * 2,
                                attention_mask=[[1] * 6] * 2,
                                offset_mapping=[[(0, 0)] * 3 + [(0, 5), (6, 10), (0, 0)],
                                                [(0, 0)] * 3 + [(6, 10), (11, 16), (0, 0)]])
        item = dict(example_id='x', question='q', context='alpha beta gamma',
                    answers=[dict(answer_start=6, text='beta gamma')])
        features, coverage = make_features([item], Tokenizer())
        self.assertEqual((features[0]['start_positions'], features[0]['end_positions']), (0, 0))
        self.assertEqual((features[1]['start_positions'], features[1]['end_positions']), (3, 4))
        self.assertEqual(coverage[0]['covered_references'], 1)
        item['answers'] = [dict(answer_start=5, text=' beta gamma')]
        features, coverage = make_features([item], Tokenizer())
        self.assertEqual((features[1]['start_positions'], features[1]['end_positions']), (3, 4))


if __name__ == '__main__':
    unittest.main()
