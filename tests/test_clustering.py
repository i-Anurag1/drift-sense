import unittest
import numpy as np

from drift_sense.matcher import _cluster_votes


class TestClustering(unittest.TestCase):
    def test_single_vote_forms_one_cluster(self):
        clusters = _cluster_votes([(10, 10, 0.9)], radius=5)
        self.assertEqual(len(clusters), 1)
        self.assertAlmostEqual(clusters[0]["cx"], 10)
        self.assertAlmostEqual(clusters[0]["weight"], 0.9)

    def test_nearby_votes_merge(self):
        votes = [(10, 10, 0.8), (12, 11, 0.7), (11, 9, 0.6)]
        clusters = _cluster_votes(votes, radius=5)
        self.assertEqual(len(clusters), 1)
        self.assertAlmostEqual(clusters[0]["weight"], 2.1)

    def test_far_votes_stay_separate(self):
        votes = [(10, 10, 0.8), (500, 500, 0.9)]
        clusters = _cluster_votes(votes, radius=5)
        self.assertEqual(len(clusters), 2)

    def test_cluster_centroid_is_weighted_average(self):
        votes = [(0, 0, 1.0), (10, 0, 3.0)]
        clusters = _cluster_votes(votes, radius=20)
        self.assertEqual(len(clusters), 1)
        # weighted average: (0*1 + 10*3) / 4 = 7.5
        self.assertAlmostEqual(clusters[0]["cx"], 7.5)

    def test_empty_votes_returns_empty(self):
        clusters = _cluster_votes([], radius=5)
        self.assertEqual(clusters, [])


if __name__ == "__main__":
    unittest.main()
