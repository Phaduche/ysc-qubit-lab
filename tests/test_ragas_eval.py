"""Temporary simulated fixtures verify evaluation logic, never research results."""
import csv
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ragas_eval as e


class RagasFormulaTests(unittest.TestCase):
    def test_paper_ratios_and_cosine(self):
        t = {"claims": {"claims": ["A", "B"]}, "support": {"verdicts": [{"supported": True}, {"supported": False}]}, "questions": {"noncommittal": False}, "similarities": {"cosines": [1, .5, 0]}, "context": {"sentence_ids": [1]}}
        self.assertEqual(e.scores(t, 4), {"faithfulness": .5, "answer_relevance": .5, "answer_relevance_raw_cosine": .5, "context_relevance": .25})
        self.assertAlmostEqual(e.cosine([1., 0.], [0., 1.]), 0)
        self.assertAlmostEqual(e.cosine([1., 0.], [-1., 0.]), -1)
        with self.assertRaises(ValueError): e.cosine([0., 0.], [1., 0.])

    def test_missing_and_refusal_verdict_require_review_not_zero(self):
        self.assertTrue(all(v is None for v in e.scores({}, 0).values()))
        self.assertIsNone(e.scores({"claims": {"claims": []}, "support": {"verdicts": []}}, 1)["faithfulness"])
        s = e.scores({"questions": {"noncommittal": True}, "similarities": {"cosines": [.8, .9, 1]}}, 1)
        self.assertIsNone(s["answer_relevance"])
        self.assertAlmostEqual(s["answer_relevance_raw_cosine"], .9)

    def test_parser_rejects_semantic_and_schema_errors(self):
        for obj in [
            {"sentence_ids": [True], "insufficient": False, "reason": "bad boolean ID"},
            {"sentence_ids": [3], "insufficient": False, "reason": "out of range"},
            {"sentence_ids": [1, 1], "insufficient": False, "reason": "duplicate"},
            {"sentence_ids": [], "insufficient": False, "reason": "conflict"},
        ]:
            with self.assertRaises(ValueError): e.parse_payload("context", json.dumps(obj), sentence_count=2)
        with self.assertRaises(ValueError): e.parse_payload("claims", '```json\n{}\n```')
        with self.assertRaises(ValueError): e.parse_payload("claims", '{"claims":["A"],"reason":"ok","extra":1}')
        with self.assertRaises(ValueError): e.parse_payload("support", '{"verdicts":[],"reason":"skipped"}', claim_count=1)
        with self.assertRaises(ValueError): e.parse_payload("questions", '{"questions":["A?"],"noncommittal":false,"reason":"ok"}')

    def test_all_chunk_sentences_count_including_overlap(self):
        ss = e.split_context([{"text": "A. B.", "citation_id": "c0"}, {"text": "B. C.", "citation_id": "c1"}])
        self.assertEqual(len(ss), 4)
        self.assertEqual([s["text"] for s in ss], ["A.", "B.", "B.", "C."])


class RagasRestartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.source = self.root/"source.sqlite"; self.db = self.root/"eval.sqlite"
        judge = {"kind": "openai_compatible", "base_url": "http://fixture.invalid", "model": "simulated-test-judge", "supports_seed": True}
        embedding = {"kind": "openai_compatible", "base_url": "http://fixture.invalid", "model": "simulated-test-embedding"}
        self.frozen = {"config": {"model": judge, "retrieval": {"embedding": embedding}}, "dataset": {"documents": [{"id": "d", "questions": [{"id": "q", "question": "When?"}]}]}}
        c = sqlite3.connect(self.source); c.executescript("CREATE TABLE manifests(id TEXT,body TEXT); CREATE TABLE jobs(id TEXT,spec TEXT,result TEXT,experiment TEXT,status TEXT);")
        c.execute("INSERT INTO manifests VALUES ('m',?)", (json.dumps(self.frozen),))
        for store in ("original", "summary"):
            s = {"source_id": "d", "question_id": "q", "repeat": 0, "run_id": "run", "store": store, "prompt_id": "P2" if store == "summary" else "", "stage": 1 if store == "summary" else 0, "seed": 7}
            r = {"data_class": "actual_model_inference", "dataset_kind": "simulation_unit_test", "output": "It happened in 2024.", "model_actual": "simulated-test-answer", "retrieved": [{"text": "It happened in 2024. It was important.", "source_id": "d", "citation_id": "c0"}]}
            c.execute("INSERT INTO jobs VALUES (?,?,?,'RAG','done')", (store, json.dumps(s), json.dumps(r)))
        c.commit(); c.close()

    def tearDown(self): self.tmp.cleanup()

    @staticmethod
    def generated(prompt, seed, **kw):
        if "Decompose every factual" in prompt: p = {"claims": ["It happened in 2024."], "reason": "one factual claim"}
        elif "For EACH numbered claim" in prompt: p = {"verdicts": [{"claim_id": 1, "supported": True, "reason": "context states 2024"}], "reason": "checked all claims"}
        elif "Using ONLY the answer" in prompt: p = {"questions": ["When?", "What year?", "When did it happen?"], "noncommittal": False, "reason": "answer gives date"}
        else: p = {"sentence_ids": [1], "insufficient": False, "reason": "first sentence answers the question"}
        return {"output": json.dumps(p), "finish_reason": "stop", "data_class": "simulation_unit_test", "input_tokens": 20, "output_tokens": 10, "total_tokens": 30}

    def test_resume_no_duplicate_and_source_unchanged(self):
        before = self.source.read_bytes()
        self.assertEqual(e.initialize(self.source, self.db, 1)["frozen_answers"], 2)
        with patch.object(e.Backend, "generate", side_effect=self.generated), patch.object(e.Embedder, "vectors", return_value=([[1., 0.]]*4, {"model": "simulation_unit_test", "tokens": 9})):
            e.execute(self.db, 3)
            e.execute(self.db)
            e.execute(self.db)
        c = e.connect(self.db)
        self.assertEqual(c.execute("SELECT count(*) FROM eval_attempts").fetchone()[0], 10)
        self.assertEqual(c.execute("SELECT count(*) FROM eval_tasks WHERE status='done'").fetchone()[0], 10); c.close()
        result = e.export_csv(self.db, self.root/"csv")
        self.assertEqual(result["evaluation_rows_done"], 2)
        self.assertEqual(result["generation_trials_added"], 0)
        self.assertEqual(self.source.read_bytes(), before)

    def test_failed_raw_response_preserved_and_no_score(self):
        e.initialize(self.source, self.db, 1)
        bad = {"output": "{}", "finish_reason": "stop", "data_class": "simulation_unit_test"}
        with patch.object(e.Backend, "generate", return_value=bad): e.execute(self.db, 1)
        c = e.connect(self.db)
        row = c.execute("SELECT status,raw_response,payload FROM eval_attempts").fetchone()
        self.assertEqual(row["status"], "failed"); self.assertEqual(json.loads(row["raw_response"])["output"], "{}"); self.assertIsNone(row["payload"])
        c.close(); e.export_csv(self.db, self.root/"csv")
        with (self.root/"csv/ragas_metrics.csv").open(encoding="utf-8-sig") as f: r = next(csv.DictReader(f))
        self.assertEqual(r["faithfulness"], ""); self.assertEqual(r["faithfulness_status"], "missing")

    def test_interruption_and_retry_retains_both_attempts(self):
        e.initialize(self.source, self.db, 1); c = e.connect(self.db)
        tid = c.execute("SELECT id FROM eval_tasks ORDER BY rowid").fetchone()[0]
        c.execute("UPDATE eval_tasks SET status='running',attempts=1 WHERE id=?", (tid,))
        c.execute("INSERT INTO eval_attempts(task_id,started,status) VALUES (?,'fixture','running')", (tid,)); c.commit(); c.close()
        with patch.object(e.Backend, "generate", side_effect=self.generated): e.execute(self.db, 1)
        c = e.connect(self.db)
        self.assertEqual([r[0] for r in c.execute("SELECT status FROM eval_attempts ORDER BY id")], ["interrupted", "done"]); c.close()

    def test_separate_db_guard_and_configuration_freeze(self):
        with self.assertRaises(ValueError): e.initialize(self.source, self.source, 1)
        e.initialize(self.source, self.db, 1)
        with self.assertRaises(ValueError): e.initialize(self.source, self.db, 1, judge={"kind": "openai_compatible", "model": "different"})


if __name__ == "__main__": unittest.main(verbosity=2)
