"""Auditable BioASQ preparation and scoring; no model downloads or training."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import string


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize_key(text):
    return ' '.join(text.casefold().split())


def load_rows(path):
    """Keep IDs and raw annotations; fail rather than silently repair bad offsets."""
    data = json.loads(Path(path).read_text())
    rows = []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                qid = qa['id']
                # This dataset uses a 24-hex BioASQ ID followed by a numeric suffix.
                match = re.fullmatch(r'([0-9a-f]{24})_\d+', qid)
                if not match:
                    raise ValueError(f'Unrecognized expanded question ID: {qid}')
                for answer in qa['answers']:
                    text, start = answer['text'], answer['answer_start']
                    if not text.strip() or start < 0 or paragraph['context'][start:start + len(text)] != text:
                        raise ValueError(f'Invalid annotation: {qid}, {answer}')
                    rows.append(dict(row_index=len(rows), expanded_id=qid,
                                     question_id=match.group(1), question=qa['question'],
                                     context=paragraph['context'], answer_text=text,
                                     answer_start=start))
    if not rows:
        raise ValueError('No answer annotations found')
    return rows


def merge_examples(rows):
    """One exact question/context pair; retain every annotated reference and source ID."""
    grouped = {}
    for row in rows:
        key = (row['question'], row['context'])
        if key not in grouped:
            grouped[key] = dict(example_id=digest(key), question=key[0], context=key[1],
                                question_ids=set(), source_ids=set(), answers=set())
        item = grouped[key]
        item['question_ids'].add(row['question_id'])
        item['source_ids'].add(row['expanded_id'])
        item['answers'].add((row['answer_start'], row['answer_text']))
    examples = []
    for item in grouped.values():
        item['question_ids'] = sorted(item['question_ids'])
        item['source_ids'] = sorted(item['source_ids'])
        item['answers'] = [dict(answer_start=s, text=t) for s, t in sorted(item['answers'])]
        examples.append(item)
    return examples


def split_examples(examples, seed=42):
    """Split connected groups: question ID, normalized question, or normalized context.

    Exact context matches are guarded; this does not detect paraphrases or partial
    document overlap. Fractions apply to connected groups, not expanded rows.
    """
    parent = list(range(len(examples)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, item in enumerate(examples):
        keys = [('id', qid) for qid in item['question_ids']]
        keys += [('q', normalize_key(item['question'])), ('c', normalize_key(item['context']))]
        for key in keys:
            if key in owners:
                parent[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = defaultdict(list)
    for i, item in enumerate(examples):
        groups[find(i)].append(item)
    stable_groups = []
    for items in groups.values():
        group_id = digest(sorted(x['example_id'] for x in items))
        stable_groups.append((group_id, items))
    stable_groups.sort()
    random.Random(seed).shuffle(stable_groups)
    count = len(stable_groups)
    if count < 10:
        raise ValueError('At least 10 independent components required for a 70/10/20 split')
    ntest, nval = math.ceil(count * .2), math.ceil(count * .1)
    splits = {'train': [], 'validation': [], 'test': []}
    for i, (group_id, items) in enumerate(stable_groups):
        split = 'test' if i < ntest else 'validation' if i < ntest + nval else 'train'
        splits[split].extend(dict(item, group_id=group_id) for item in items)
    assert_independent(splits)
    return splits


def assert_independent(splits):
    owners = {}
    for split, items in splits.items():
        for item in items:
            keys = [('id', qid) for qid in item['question_ids']]
            keys += [('q', normalize_key(item['question'])), ('c', normalize_key(item['context']))]
            for key in keys:
                if key in owners and owners[key] != split:
                    raise ValueError(f'Cross-split overlap: {key[0]}')
                owners[key] = split


def normalize_answer(text):
    """SQuAD-style English normalization, stated explicitly for this extractive task."""
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def score_answer(prediction, references):
    """Score against original nonempty references, taking the best valid alternative."""
    if not references or any(not text.strip() for text in references):
        raise ValueError('Factoid references must be nonempty original annotations')
    pred = normalize_answer(prediction)
    em, f1 = [], []
    for reference in references:
        gold = normalize_answer(reference)
        em.append(float(pred == gold))
        pt, gt = pred.split(), gold.split()
        overlap = sum((Counter(pt) & Counter(gt)).values())
        f1.append(float(pt == gt) if not pt or not gt else 2 * overlap / (len(pt) + len(gt)))
    return {'em': max(em), 'f1': max(f1)}


def best_span(start_logits, end_logits, offsets, context, max_answer_tokens=30):
    """Exact constrained search over context tokens; offsets for other tokens are None."""
    if not (len(start_logits) == len(end_logits) == len(offsets)) or max_answer_tokens < 1:
        raise ValueError('Invalid span inputs')
    best = None
    for start, offset in enumerate(offsets):
        if offset is None or offset[1] <= offset[0]:
            continue
        for end in range(start, min(len(offsets), start + max_answer_tokens)):
            end_offset = offsets[end]
            if end_offset is None or end_offset[1] <= end_offset[0]:
                break
            if end_offset[1] <= offset[0]:
                continue
            score = float(start_logits[start]) + float(end_logits[end])
            if not math.isfinite(score):
                continue
            if best is None or score > best['score']:
                best = dict(text=context[offset[0]:end_offset[1]], score=score,
                            char_start=offset[0], char_end=end_offset[1])
    return best


def make_features(examples, tokenizer, max_length=384, stride=128):
    """Overlapping context windows; both labels use CLS when a window lacks an answer.

    All windows are retained. Evaluation merges windows back to one prediction per
    question/context example and uses original references, never the CLS labels.
    """
    features, coverage = [], []
    for item in examples:
        encoded = tokenizer(item['question'], item['context'], truncation='only_second',
                            max_length=max_length, stride=stride, padding='max_length',
                            return_offsets_mapping=True, return_overflowing_tokens=True)
        covered = set()
        answer_lengths = []
        for window, ids in enumerate(encoded['input_ids']):
            sequence = encoded.sequence_ids(window)
            offsets = [tuple(o) if seq == 1 and o[1] > o[0] else None
                       for o, seq in zip(encoded['offset_mapping'][window], sequence)]
            context_tokens = [i for i, offset in enumerate(offsets) if offset is not None]
            cls = ids.index(tokenizer.cls_token_id)
            start = end = cls
            visible = []
            for ai, answer in enumerate(item['answers']):
                # WordPiece offsets omit whitespace. Keep the original reference
                # verbatim for scoring, but align its non-whitespace bounds.
                raw = answer['text']
                a = answer['answer_start'] + len(raw) - len(raw.lstrip())
                b = answer['answer_start'] + len(raw.rstrip())
                if context_tokens and offsets[context_tokens[0]][0] <= a and offsets[context_tokens[-1]][1] >= b:
                    starts = [i for i in context_tokens if offsets[i][0] <= a < offsets[i][1]]
                    ends = [i for i in context_tokens if offsets[i][0] < b <= offsets[i][1]]
                    if starts and ends and starts[0] <= ends[-1]:
                        visible.append((starts[0], ends[-1]))
                        answer_lengths.append(ends[-1] - starts[0] + 1)
                        covered.add(ai)
            if visible:
                start, end = visible[0]
            feature = {key: encoded[key][window] for key in ['input_ids', 'attention_mask', 'token_type_ids'] if key in encoded}
            feature.update(start_positions=start, end_positions=end, offsets=offsets,
                           example_id=item['example_id'])
            features.append(feature)
        coverage.append(dict(example_id=item['example_id'], references=len(item['answers']),
                             covered_references=len(covered),
                             shortest_answer_tokens=min(answer_lengths) if answer_lengths else None))
        if not covered:
            raise ValueError(f'No window contains a complete annotated answer: {item["example_id"]}')
    return features, coverage
