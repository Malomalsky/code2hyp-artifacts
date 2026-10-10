"""Small tests of the new selection boundary; existing transforms stay frozen."""

import unittest
from root_tfidf_external_20261010 import cf_id, choose


class SelectionTests(unittest.TestCase):
    def test_urls(self):
        self.assertEqual(cf_id("https://codeforces.com/contest/123/problem/A"), "123:A")
        self.assertEqual(cf_id("https://codeforces.com/problemset/problem/123/A"), "123:A")
        self.assertIsNone(cf_id("https://other.example/contest/123/problem/A"))
        self.assertIsNone(cf_id("https://atcoder.jp/contests/abc/tasks/a"))

    def test_deterministic_disjoint_selection(self):
        rows = [{"cluster_id": p, "d0": f"{p}:{i}", "d1": f"t{p}:{i}", "d2": f"a{p}:{i}"}
                for p in ("1:A", "2:B") for i in range(18)]
        selected, eligible, removed = choose(rows, 2)
        self.assertEqual((eligible, len(selected), dict(removed)), (2, 32, {}))
        self.assertEqual(choose(rows[::-1], 2)[0], selected)
        for p in ("1:A", "2:B"):
            self.assertEqual(sum(r["cluster_id"] == p and r["role"] == "query" for r in selected), 8)
        contaminated = rows + [{**rows[0], "cluster_id": "3:C"}, dict(rows[-1])]
        _, _, removed = choose(contaminated, 2)
        self.assertEqual(removed["cross_problem_duplicate"], 2)
        self.assertEqual(removed["within_problem_duplicate"], 1)
        self.assertEqual(choose(rows[:15], 1)[0], [])


if __name__ == "__main__":
    unittest.main()
