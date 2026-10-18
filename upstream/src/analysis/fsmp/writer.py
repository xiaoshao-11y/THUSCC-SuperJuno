import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threading import Thread
from queue import Queue
from scipy.sparse import coo_array

class BufferedParquetWriter:
    """Buffered writer.

    It saves the tables and flush to file if the number of rows is larger than a
    specified number. Therefore the output parquet file will contains lesser row groups.
    """

    def __init__(self, path: str, schema: pa.Schema, row_group_limit: int = 1048576):
        """
        path: str
            Output file name.
        schema: Schema
            Output parquet table schema.
        row_group_limit: int
            The limit size of the buffer.
        """
        self.writer = pq.ParquetWriter(path, compression="ZSTD", schema=schema)
        self.buffer = []
        self.num_row = 0
        self.row_group_limit = row_group_limit

    def flush(self):
        """Flush the buffer to the file, and reset the buffer."""
        if self.num_row != 0:
            table = pa.concat_tables(self.buffer)
            self.writer.write_table(table, row_group_size=self.num_row)
            self.buffer = []
            self.num_row = 0

    def write_table(self, table: pa.Table):
        """Write a table to the buffer."""
        self.buffer.append(table)
        self.num_row += table.num_rows
        if self.num_row >= self.row_group_limit:
            self.flush()

    def close(self):
        """Flush and close the writer."""
        self.flush()
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()
        return False


def write_daemon(filename, schema, queue, append_method):
    """Daemon for writer subprocess

    A daemon live inside the subprocess handling the parquet writer.  It
    creates the writer and buffer, listens to the data and signal from master
    process and write the data to file.

    Parameters
    ----------
    filename
        Output parquet filename
    schema
        Output parquet table schema.
    pipe
        One end of pipe generated from multiprocessing.Pipe()
    append_method
        The implementation to append data
    """
    with BufferedParquetWriter(filename, schema=schema) as writer:
        while True:
            data = queue.get()
            if data == "TERMINATE":
                break
            append_method(writer, schema, *data)


class WriterBase:
    def __init__(self, opt: str, schema: pa.Schema, append_method):
        """
        opt: str
            Output file name.
        schema: Schema
            Output parquet table schema.
        append_method
            the staticmethod `append` of derived class
        """
        self.schema = schema
        self.opt = opt
        self.append = append_method
        self.queue = Queue()
        self.process = Thread(
            target=write_daemon,
            args=(self.opt, self.schema, self.queue, self.append),
        )
        self.process.start()

    def append_async(self, *data):
        self.queue.put(data)

    def close(self):
        self.queue.put("TERMINATE")
        self.process.join()

class BasicWriter(WriterBase):
    def __init__(
        self,
        opt: str,
        eid_dtype: np.dtype = np.dtype(np.uint32),
        ch_dtype: np.dtype = np.dtype(np.uint16),
        offset_dtype: np.dtype = np.dtype(np.uint16),
    ):
        """
        opt: str
            Output file name.
        """
        schema = pa.schema(
            [
                pa.field("eid", pa.from_numpy_dtype(eid_dtype)),
                pa.field("ch", pa.from_numpy_dtype(ch_dtype)),
                pa.field("offset", pa.from_numpy_dtype(offset_dtype)),
                pa.field("step", pa.uint16()),
                pa.field("flip", pa.int8()),
                pa.field("s0", pa.uint16()),
                pa.field("op", pa.uint16()),
                pa.field("time", pa.float64()),
                pa.field("e", pa.int8()),
                pa.field("delta_nu", pa.float64()),
                pa.field("nu_lc", pa.float64()),
                pa.field("count", pa.uint16()),
            ]
        )
        super().__init__(opt, schema, self.append)

    @staticmethod
    def append(writer, schema, TRIALS, eid, ch, offset, init_s0, nu_lc0, ind, flip, s0, op, time, e, delta_nu, nu_lc):
        """
        eid: (l_e,)
        ch: (l_e,)
        offset: (l_e,)
        init_s0: (l_e,)
        flip: (l_e * TRIALS,)
        s0: (l_e * TRIALS,)
        op: (l_e * TRIALS,)
        time: (l_e * TRIALS,)
        e: (l_e * TRIALS,)
        delta_nu: (l_e * TRIALS,)
        nu_lc: (l_e * TRIALS+1,)
        """
        # 把稀疏的历史记录紧凑化
        l_e = len(eid)
        l_loc = (l_e, ) + tuple(map(lambda x: len(x), ind))
        loc = np.arange(0, TRIALS+1, dtype=np.uint16()) # sparse 0, 1, ... TRIALS
        cat_loc = np.repeat(loc, l_loc)
        step = loc - 1
        step[0] = 0 # 0, 0, 1, 2, 3, ..., TRIALS-1
        cat_step = np.repeat(step, l_loc)
        ind.insert(0, np.arange(l_e))
        ind_flat = np.concatenate(ind)

        sparse = {}
        flip.insert(0, np.int8(0))
        for k, v in zip (("step", "flip"),
                        (cat_step, np.repeat(flip, l_loc))):
            sparse[k] = coo_array((v, (ind_flat, cat_loc))).tocsr().data

        for k, v in zip(("s0", "op", "time", "e", "delta_nu", "nu_lc"),
                           (s0, op, time, e, delta_nu, nu_lc)):
            if k == "s0":
                v.insert(0, init_s0)
            elif k == "nu_lc":
                v.insert(0, nu_lc0)
            else: # place holders
                v.insert(0, np.zeros(l_e, dtype=v[0].dtype))
            sparse[k] = coo_array((np.concatenate(v), (ind_flat, cat_loc))).tocsr().data

        loca = np.append(coo_array((cat_loc, (ind_flat, cat_loc))).tocsr().data, 0)
        (ind_chain,) = np.nonzero(loca == 0)
        l_chain = np.diff(ind_chain)
        for k, v in zip(("eid", "ch", "offset"),
                       (eid, ch, offset)):
            sparse[k] = np.repeat(v, l_chain)

        l_chain[-1] += 1
        loca[:-1] = sparse["step"] # 考虑 step 与 loc 的不同来记 count
        loca += np.repeat(np.arange(len(l_chain)) * TRIALS, l_chain)
        loca[-1] += TRIALS
        sparse["count"] = np.diff(loca)

        writer.write_table(pa.Table.from_pydict(sparse, schema))

class FullWriter(WriterBase):
    def __init__(
        self,
        opt: str,
        eid_dtype: np.dtype = np.dtype(np.uint32),
        ch_dtype: np.dtype = np.dtype(np.uint16),
        offset_dtype: np.dtype = np.dtype(np.uint16),
    ):
        """
        opt: str
            Output file name.
        """
        schema = pa.schema(
            [
                pa.field("eid", pa.from_numpy_dtype(eid_dtype)),
                pa.field("ch", pa.from_numpy_dtype(ch_dtype)),
                pa.field("offset", pa.from_numpy_dtype(offset_dtype)),
                pa.field("step", pa.uint16()),
                pa.field("PEt", pa.list_(pa.float64())),
                pa.field("e", pa.list_(pa.int8())),
                pa.field("q", pa.list_(pa.float64())),
                pa.field("nu_lc", pa.float64())
            ]
        )
        super().__init__(opt, schema, self.append)

    @staticmethod
    def append(writer, schema, index, tq, i_loop, NPE, s, e, q, nu_lc):
        """
        index: (l_e,)
            Coefficients from sparsify file.
        i_loop: int
            Current step.
        NPE: (l_e,)
            NPEs.
        s: (l_e, l_s)
            PEs.
        e: (l_e, l_s)
            MCPs.
        cs: cp.ndarray (l_e, l_s, l_wave)
            Interpolated cx.
        z: cp.ndarray (l_e, l_wave)
            z.
        """
        # MCMC times are indices into the tq grid, whose origin may be negative.
        time_origin = tq["t_s"][:, 0]
        l_e = len(index)
        s_full = []
        e_full = []
        q_full = []
        for i, npe in enumerate(NPE):
            npe = int(npe)
            s_full.append((s[i, :npe] + time_origin[i]).astype(np.float64, copy=False))
            e_full.append(e[i, :npe].astype(np.int8, copy=False))
            if q is None:
                q_full.append([None] * npe)
            else:
                q_full.append(q[i, :npe].astype(np.float64, copy=False))
        table = pa.Table.from_arrays(
            [
                index["eid"],
                index["ch"],
                index["offset"],
                np.full(l_e, i_loop, dtype=np.uint16),
                pa.array(s_full, type=schema.field("PEt").type),
                pa.array(e_full, type=schema.field("e").type),
                pa.array(q_full, type=schema.field("q").type),
                nu_lc,
            ],
            schema=schema,
        )
        writer.write_table(table)
