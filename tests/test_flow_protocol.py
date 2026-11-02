"""Diagram contract tests. All generated responses here are isolated test fixtures."""
import collections
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ysc


def fixture_response(output):
    return {
        'output': output, 'finish_reason': 'stop',
        'model_actual': 'isolated-protocol-test-fixture',
        'input_tokens': None, 'output_tokens': None, 'total_tokens': None,
        'token_method': 'unavailable', 'seed_sent': False, 'raw': {},
        'data_class': 'simulation_fixture',
    }


class FlowProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ysc.load(ROOT / 'configs/diagram_3repeats.json')
        self.data = ysc.load(ROOT / 'data/supplied_smoke.json')
        self.frozen, self.jobs = ysc.plan(self.cfg, self.data)
        self.by_id = {s['id']: s for s in self.jobs}
        self.by_doc = {d['id']: d for d in self.data['documents']}
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / 'fixtures.sqlite'
        db = ysc.connect(self.db)
        db.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_three_repeat_config_changes_only_repeat_count(self):
        original = ysc.load(ROOT / 'configs/local_smoke.json')
        expected = copy.deepcopy(original)
        expected['repeats'] = 3
        self.assertEqual(self.cfg, expected)
        self.assertEqual(self.cfg['depth'], 5)
        self.assertEqual(self.cfg['summary_stages'], [1, 5])

    def test_exact_198_job_plan_and_three_prompt_conditions(self):
        self.assertEqual(len(self.data['documents']), 2)
        self.assertEqual(sum(len(d['questions']) for d in self.data['documents']), 4)
        self.assertEqual(collections.Counter(s['experiment'] for s in self.jobs),
                         {'A': 6, 'B': 18, 'C': 90, 'RAG': 84})
        self.assertEqual(len(self.jobs), 198)
        self.assertEqual(len(self.by_id), 198)
        self.assertEqual(set(ysc.PROMPTS), {'P1', 'P2', 'P3'})
        for exp in ['A', 'B', 'C']:
            group = [s for s in self.jobs if s['experiment'] == exp]
            self.assertEqual({s['prompt_id'] for s in group}, set(ysc.PROMPTS))
        a = [s for s in self.jobs if s['experiment'] == 'A']
        self.assertTrue(all(s['repeat'] == 0 and s['stage'] == 1 for s in a))

    def test_a_b_and_c_first_step_are_independent_h0_inputs(self):
        jobs = [s for s in self.jobs if s['experiment'] in ['A', 'B']
                or (s['experiment'] == 'C' and s['stage'] == 1)]
        self.assertEqual(len(jobs), 42)
        for s in jobs:
            with self.subTest(experiment=s['experiment'], prompt=s['prompt_id'],
                              repeat=s['repeat'], source=s['source_id']):
                self.assertEqual(s['dependencies'], [])
                with patch.object(ysc.Backend, 'generate',
                                  return_value=fixture_response('isolated fixture')) as generate:
                    result = ysc.run_job(self.db, s, self.frozen)
                expected = self.by_doc[s['source_id']]['text']
                args = generate.call_args.args
                self.assertEqual(args[2], expected)
                self.assertEqual(args[0], ysc.PROMPTS[s['prompt_id']] + '\n자료:\n' + expected)
                self.assertEqual(result['input_text_hash'], ysc.digest(expected))

    def test_c_dependencies_stay_inside_one_chain(self):
        for s in [s for s in self.jobs if s['experiment'] == 'C']:
            with self.subTest(prompt=s['prompt_id'], repeat=s['repeat'], stage=s['stage']):
                if s['stage'] == 1:
                    self.assertEqual(s['dependencies'], [])
                    continue
                self.assertEqual(len(s['dependencies']), 1)
                previous = self.by_id[s['dependencies'][0]]
                for key in ['experiment', 'source_id', 'prompt_id', 'repeat', 'seed']:
                    self.assertEqual(previous[key], s[key])
                self.assertEqual(previous['stage'], s['stage'] - 1)

    def test_c_execution_passes_only_the_previous_output(self):
        outputs = {}
        for s in [s for s in self.jobs if s['experiment'] == 'C']:
            expected = (self.by_doc[s['source_id']]['text'] if s['stage'] == 1
                        else outputs[s['dependencies'][0]])
            output = 'isolated chain fixture: ' + s['id']
            with patch.object(ysc.Backend, 'generate',
                              return_value=fixture_response(output)) as generate:
                result = ysc.run_job(self.db, s, self.frozen)
            self.assertEqual(generate.call_args.args[2], expected)
            self.assertEqual(result['input_text_hash'], ysc.digest(expected))
            outputs[s['id']] = output
            db = ysc.connect(self.db)
            db.execute('INSERT INTO jobs(id,experiment,spec,status,result) VALUES (?,?,?,?,?)',
                       (s['id'], 'C', ysc.dumps(s), 'done', ysc.dumps(result)))
            db.commit()
            db.close()

    def test_rag_summary_store_uses_all_documents_from_same_condition(self):
        docs = set(self.by_doc)
        rag = [s for s in self.jobs if s['experiment'] == 'RAG']
        self.assertEqual(sum(s['store'] == 'original' for s in rag), 12)
        self.assertEqual(sum(s['store'] == 'summary' for s in rag), 72)
        for s in rag:
            if s['store'] == 'original':
                self.assertEqual(s['dependencies'], [])
                self.assertEqual((s['prompt_id'], s['stage']), ('', 0))
                continue
            dependencies = [self.by_id[jid] for jid in s['dependencies']]
            self.assertEqual(len(dependencies), len(docs))
            self.assertEqual({d['source_id'] for d in dependencies}, docs)
            for dep in dependencies:
                self.assertEqual(dep['experiment'], 'C')
                for key in ['prompt_id', 'repeat', 'stage', 'seed']:
                    self.assertEqual(dep[key], s[key])

    def test_rag_execution_searches_complete_selected_store(self):
        summary_jobs = [s for s in self.jobs if s['experiment'] == 'C']
        expected_summary = {s['id']: 'isolated summary fixture ' + s['id'] for s in summary_jobs}
        db = ysc.connect(self.db)
        for s in summary_jobs:
            db.execute('INSERT INTO jobs(id,experiment,spec,status,result) VALUES (?,?,?,?,?)',
                       (s['id'], 'C', ysc.dumps(s), 'done',
                        ysc.dumps(fixture_response(expected_summary[s['id']]))))
        db.commit()
        db.close()
        for s in [s for s in self.jobs if s['experiment'] == 'RAG']:
            expected = ([{'id': d['id'], 'text': d['text']} for d in self.data['documents']]
                        if s['store'] == 'original' else
                        [{'id': self.by_id[jid]['source_id'], 'text': expected_summary[jid]}
                         for jid in s['dependencies']])
            hits = [{'source_id': expected[0]['id'], 'text': expected[0]['text'],
                     'start': 0, 'end': len(expected[0]['text']), 'score': 1.0,
                     'citation_id': 'c0'}]
            with patch.object(ysc, 'retrieve', return_value=(hits, 'isolated-index')) as retrieve, \
                 patch.object(ysc.Backend, 'generate',
                              return_value=fixture_response('isolated answer [c0]')) as generate:
                result = ysc.run_job(self.db, s, self.frozen)
            self.assertEqual(retrieve.call_args.args[2], expected)
            self.assertEqual(generate.call_args.args[2], '[c0] ' + expected[0]['text'])
            self.assertTrue(generate.call_args.args[3])
            self.assertEqual(result['data_class'], 'simulation_fixture')

    def test_seed_is_paired_across_prompts_and_stores_and_fixed_within_chain(self):
        for s in self.jobs:
            self.assertEqual(s['seed'], self.cfg['seed'] + s['repeat'])
        for repeat in range(3):
            seeds = {s['seed'] for s in self.jobs if s['repeat'] == repeat}
            self.assertEqual(seeds, {1729 + repeat})
        # Replicate zero deliberately shares A/B/C-H1 conditions; it is not a new source sample.
        for doc in self.by_doc:
            for prompt in ysc.PROMPTS:
                first = [s for s in self.jobs if s['source_id'] == doc and s['prompt_id'] == prompt
                         and s['repeat'] == 0 and s['stage'] == 1
                         and s['experiment'] in ['A', 'B', 'C']]
                self.assertEqual(len(first), 3)
                self.assertEqual({s['seed'] for s in first}, {1729})
                self.assertTrue(all(not s['dependencies'] for s in first))


if __name__ == '__main__':
    unittest.main(verbosity=2)
