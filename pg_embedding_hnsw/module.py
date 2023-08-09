import subprocess
import sys

import psycopg

from ..base.module import BaseANN

class PGEmbedding(BaseANN):
    def __init__(self, metric, lists):
        self._metric = metric
        self._lists = lists
        self._cur = None

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def fit(self, X):
        subprocess.run("service postgresql start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        conn = psycopg.connect(user="ann", password="ann", dbname="ann", autocommit=True)
        cur = conn.cursor()
        cur.execute("CREATE TABLE items (id int, embedding real[])")
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        print("copying data...")
        with cur.copy("COPY items (id, embedding) FROM STDIN") as copy:
            for i, embedding in enumerate(X):
#               copy.write_row((i, embedding))
                copy.write_row((i, "{" + ",".join([str(x) for x in embedding]) + "}"))
        print("creating index...")
        if self._metric == "angular":
            cur.execute(
                "CREATE INDEX hnsw_idx ON items USING hnsw (embedding ann_cos_ops) WITH (dims=1536, m = %d, efConstruction = 16)"
                % self._lists
            )
        elif self._metric == "euclidean":
            cur.execute("CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = %d)" % self._lists)
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("vacuum and checkpoint")
        cur.execute("VACUUM ANALYZE items;")
        print("warm cache")
        cur.execute("SELECT pg_prewarm('items')")
        cur.execute("SELECT pg_prewarm('hnsw_idx')")
        print("done!")
        self._cur = cur

    def set_query_arguments(self, probes):
        self._probes = probes
        self._cur.execute("ALTER INDEX hnsw_idx SET ( efSearch = %d )" % probes)
        self._cur.execute("SET enable_seqscan = off")
        # TODO set based on available memory
        self._cur.execute("SET work_mem = '8GB'")
        # disable parallel query execution
        self._cur.execute("SET max_parallel_workers_per_gather = 0")

    def query(self, v, n):
#       self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        self._cur.execute(self._query, ("{" + ",".join([str(x) for x in v]) + "}", n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        self._cur.execute("SELECT pg_relation_size('hnsw_idx')")
        return self._cur.fetchone()[0] / 1024

    def __str__(self):
        return f"PGVector(lists={self._lists}, probes={self._probes})"
