#!/usr/bin/env python3

import sqlite3
import struct
import unittest

import benchmark_row_addressed_route_sqlite as benchmark


class RowAddressedRouteSQLiteTests(unittest.TestCase):
    def test_route_slice_and_unpack_respect_fixed_capacity(self):
        first = struct.pack(
            f"<{benchmark.ROUTE_CAPACITY + 1}I",
            2,
            7,
            9,
            *([0] * (benchmark.ROUTE_CAPACITY - 2)),
        )
        second = struct.pack(
            f"<{benchmark.ROUTE_CAPACITY + 1}I",
            1,
            11,
            *([0] * (benchmark.ROUTE_CAPACITY - 1)),
        )
        packed = first + second
        self.assertEqual(benchmark.unpack_route(benchmark.route_blob(packed, 0)), (7, 9))
        self.assertEqual(benchmark.unpack_route(benchmark.route_blob(packed, 1)), (11,))

    def test_injected_precommit_failure_rolls_back_all_route_rows(self):
        database = sqlite3.connect(":memory:")
        benchmark.create_schema(database)
        embedding = struct.pack(f"<{benchmark.DIMENSION}f", *([0.0] * benchmark.DIMENSION))
        edge = struct.pack(f"<{benchmark.EDGE_CAPACITY + 1}I", *([0] * (benchmark.EDGE_CAPACITY + 1)))
        route = struct.pack(f"<{benchmark.ROUTE_CAPACITY + 1}I", *([0] * (benchmark.ROUTE_CAPACITY + 1)))
        with self.assertRaises(RuntimeError):
            benchmark.apply_epoch(
                database,
                [(0, 1, embedding, 2)],
                [(0, 0, edge), (1, 0, edge)],
                [(0, route)],
                [],
                1,
                failure_stage="before-commit",
            )
        self.assertEqual(
            benchmark.database_counts(database),
            {"nodes": 0, "edges": 0, "routes": 0, "anchors": 0},
        )


if __name__ == "__main__":
    unittest.main()
