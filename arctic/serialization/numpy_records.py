import logging
import numpy as np

from pandas import DataFrame, MultiIndex, Series, DatetimeIndex, Index

from arctic.store._ndarray_store import MAX_DOCUMENT_SIZE, _CHUNK_SIZE
from ..exceptions import ArcticException
try:  # 0.21+ Compatibility
    from pandas._libs.tslib import Timestamp
    from pandas._libs.tslibs.timezones import get_timezone
except ImportError:
    try:  # 0.20.x Compatibility
        from pandas._libs.tslib import Timestamp, get_timezone
    except ImportError:  # <= 0.19 Compatibility
        from pandas.tslib import Timestamp, get_timezone


log = logging.getLogger(__name__)

DTN64_DTYPE = 'datetime64[ns]'


def _to_primitive(arr, string_max_len=None):
    if arr.dtype.hasobject:
        if len(arr) > 0:
            if isinstance(arr[0], Timestamp):
                return np.array([t.value for t in arr], dtype=DTN64_DTYPE)
        if string_max_len:
            return np.array(arr.astype('U{:d}'.format(string_max_len)))
        return np.array(list(arr))
    return arr


def _multi_index_to_records(index, empty_index):
    # array of tuples to numpy cols. copy copy copy
    if not empty_index:
        ix_vals = list(map(np.array, [index.get_level_values(i) for i in range(index.nlevels)]))
    else:
        # empty multi index has no size, create empty arrays for recarry..
        ix_vals = [np.array([]) for n in index.names]
    index_names = list(index.names)
    count = 0
    for i, n in enumerate(index_names):
        if n is None:
            index_names[i] = 'level_%d' % count
            count += 1
            log.info("Level in MultiIndex has no name, defaulting to %s" % index_names[i])
    index_tz = [get_timezone(i.tz) if isinstance(i, DatetimeIndex) else None for i in index.levels]
    return ix_vals, index_names, index_tz


class PandasSerializer(object):

    def _index_to_records(self, df):
        metadata = {}
        index = df.index
        index_tz = None

        if isinstance(index, MultiIndex):
            ix_vals, index_names, index_tz = _multi_index_to_records(index, len(df) == 0)
        else:
            ix_vals = [index.values]
            index_names = list(index.names)
            if index_names[0] is None:
                index_names = ['index']
                log.info("Index has no name, defaulting to 'index'")
            if isinstance(index, DatetimeIndex) and index.tz is not None:
                index_tz = get_timezone(index.tz)

        if index_tz is not None:
            metadata['index_tz'] = index_tz
        metadata['index'] = index_names

        return index_names, ix_vals, metadata

    def _index_from_records(self, recarr):
        index = recarr.dtype.metadata['index']

        if len(index) == 1:
            rtn = Index(np.copy(recarr[str(index[0])]), name=index[0])
            if isinstance(rtn, DatetimeIndex) and 'index_tz' in recarr.dtype.metadata:
                rtn = rtn.tz_localize('UTC').tz_convert(recarr.dtype.metadata['index_tz'])
        else:
            level_arrays = []
            index_tz = recarr.dtype.metadata.get('index_tz', [])
            for level_no, index_name in enumerate(index):
                # build each index level separately to ensure we end up with the right index dtype
                level = Index(np.copy(recarr[str(index_name)]))
                if level_no < len(index_tz):
                    tz = index_tz[level_no]
                    if tz is not None:
                        if not isinstance(level, DatetimeIndex) and len(level) == 0:
                            # index type information got lost during save as the index was empty, cast back
                            level = DatetimeIndex([], tz=tz)
                        else:
                            level = level.tz_localize('UTC').tz_convert(tz)
                level_arrays.append(level)
            rtn = MultiIndex.from_arrays(level_arrays, names=index)
        return rtn

    def _to_records(self, df, string_max_len=None):
        """
        Similar to DataFrame.to_records()
        Differences:
            Attempt type conversion for pandas columns stored as objects (e.g. strings),
            as we can only store primitives in the ndarray.
            Use dtype metadata to store column and index names.

        string_max_len: integer - enforces a string size on the dtype, if any
                                  strings exist in the record
        """

        index_names, ix_vals, metadata = self._index_to_records(df)
        columns, column_vals, multi_column = self._column_data(df)

        if "" in columns:
            raise ArcticException("Cannot use empty string as a column name.")

        if multi_column is not None:
            metadata['multi_column'] = multi_column
        metadata['columns'] = columns
        names = index_names + columns

        arrays = []
        for arr in ix_vals + column_vals:
            arrays.append(_to_primitive(arr, string_max_len))

        dtype = np.dtype([(str(x), v.dtype) if len(v.shape) == 1 else (str(x), v.dtype, v.shape[1]) for x, v in zip(names, arrays)],
                         metadata=metadata)

        # The argument names is ignored when dtype is passed
        rtn = np.rec.fromarrays(arrays, dtype=dtype, names=names)
        # For some reason the dtype metadata is lost in the line above
        # and setting rtn.dtype to dtype does not preserve the metadata
        # see https://github.com/numpy/numpy/issues/6771

        return rtn, dtype

    def can_convert_to_records_without_objects(self, df, symbol):
        # We can't easily distinguish string columns from objects
        try:
            arr, _ = self._to_records(df)
        except Exception as e:
            # This exception will also occur when we try to write the object so we fall-back to saving using Pickle
            log.info('Pandas dataframe %s caused exception "%s" when attempting to convert to records. Saving as Blob.'
                     % (symbol, repr(e)))
            return False
        else:
            if arr.dtype.hasobject:
                log.info('Pandas dataframe %s contains Objects, saving as Blob' % symbol)
                # Will fall-back to saving using Pickle
                return False
            elif any([len(x[0].shape) for x in arr.dtype.fields.values()]):
                log.info('Pandas dataframe %s contains >1 dimensional arrays, saving as Blob' % symbol)
                return False
            else:
                return True

    def serialize(self, item):
        raise NotImplementedError

    def deserialize(self, item):
        raise NotImplementedError


class SeriesSerializer(PandasSerializer):
    TYPE = 'series'

    def _column_data(self, s):
        if s.name is None:
            log.info("Series has no name, defaulting to 'values'")
        columns = [s.name if s.name else 'values']
        column_vals = [s.values]
        return columns, column_vals, None

    def deserialize(self, item):
        index = self._index_from_records(item)
        name = item.dtype.names[-1]
        return Series.from_array(item[name], index=index, name=name)

    def serialize(self, item, string_max_len=None):
        return self._to_records(item, string_max_len)


class DataFrameSerializer(PandasSerializer):
    TYPE = 'df'

    def _column_data(self, df):
        columns = list(map(str, df.columns))
        if columns != list(df.columns):
            log.info("Dataframe column names converted to strings")
        column_vals = [df[c].values for c in df.columns]

        if isinstance(df.columns, MultiIndex):
            ix_vals, ix_names, _ = _multi_index_to_records(df.columns, False)
            vals = [list(val) for val in ix_vals]
            str_vals = [list(map(str, val)) for val in ix_vals]
            if vals != str_vals:
                log.info("Dataframe column names converted to strings")
            return columns, column_vals, {"names": list(ix_names), "values": str_vals}
        else:
            return columns, column_vals, None

    def deserialize(self, item):
        index = self._index_from_records(item)
        column_fields = [x for x in item.dtype.names if x not in item.dtype.metadata['index']]
        multi_column = item.dtype.metadata.get('multi_column')
        if len(item) == 0:
            rdata = item[column_fields] if len(column_fields) > 0 else None
            if multi_column is not None:
                columns = MultiIndex.from_arrays(multi_column["values"], names=multi_column["names"])
                return DataFrame(rdata, index=index, columns=columns)
            else:
                return DataFrame(rdata, index=index)

        columns = item.dtype.metadata['columns']
        df = DataFrame(data=item[column_fields], index=index, columns=columns)

        if multi_column is not None:
            df.columns = MultiIndex.from_arrays(multi_column["values"], names=multi_column["names"])

        return df

    def serialize(self, item, string_max_len=None):
        return self._to_records(item, string_max_len)


class LazyIncrementalSerializer(object):
    def __init__(self, serializer, original_df, chunk_size=_CHUNK_SIZE, string_max_len=None):
        if chunk_size < 1:
            raise ArcticException("LazyIncrementalSerializer can't be initialized "
                                  "with a chunk_size < 1 ({})".format(chunk_size))
        if not serializer:
            raise ArcticException("LazyIncrementalSerializer can't be initialized "
                                  "with a None serializer object")
        self.original_df = original_df
        self.chunk_size = chunk_size
        self.string_max_len = string_max_len
        self._serializer = serializer
        # The state which needs to be lazily initialized
        self._dtype = None
        self._rows_per_chunk = 0
        self._initialized = False
        self._first_chunk = None

    def _lazy_init(self):
        if self._initialized:
            return

        # Serialize the first row to obtain info about row size in bytes (cache first row)
        # Also raise an Exception early, if data are not serializable
        first_chunk, dtype = self._serializer.serialize(self.original_df[0:1] if len(self) > 0 else self.original_df,
                                                        string_max_len=self.string_max_len)

        # Compute the number of rows which can fit in a chunk
        rows_per_chunk = 0
        if len(self) > 0 and self.chunk_size > 1:
            rows_per_chunk = self._calculate_rows_per_chunk(first_chunk)

        # Initialize object's state
        self._first_chunk = first_chunk
        self._dtype = dtype
        self._rows_per_chunk = rows_per_chunk
        self._initialized = True

    def _calculate_rows_per_chunk(self, first_chunk):
        sze = int(first_chunk.dtype.itemsize * np.prod(first_chunk.shape[1:]))
        sze = sze if sze < self.chunk_size else self.chunk_size
        rows_per_chunk = int(self.chunk_size / sze)
        if rows_per_chunk < 1:
            # If a row size is larger than chunk_size, use the maximum document size
            logging.warning('Chunk size of {} is too small to fit a row ({}). '
                            'Using maximum document size.'.format(self.chunk_size, MAX_DOCUMENT_SIZE))
            # For huge rows, fall-back to using a very large document size, less than max-allowed by MongoDB
            rows_per_chunk = int(MAX_DOCUMENT_SIZE / sze)
            if rows_per_chunk < 1:
                raise ArcticException("Serialization failed to split data into max sized chunks.")
        return rows_per_chunk

    def __len__(self):
        return len(self.original_df)

    @property
    def shape(self):
        return self.original_df.shape

    @property
    def dtype(self):
        self._lazy_init()
        return self._dtype

    @property
    def rows_per_chunk(self):
        self._lazy_init()
        return self._rows_per_chunk

    @property
    def generator(self):
        return self._generator()

    @property
    def generator_bytes(self):
        return self._generator(get_bytes=True)

    def _generator(self, get_bytes=False):
        self._lazy_init()

        if len(self) == 0:
            return

        # Compute the total number of chunks
        total_chunks = int(np.ceil(float(len(self)) / self._rows_per_chunk))

        # Perform serialization for each chunk
        for i in xrange(total_chunks):
            chunk, dtype = self._serializer.serialize(
                self.original_df[i * self._rows_per_chunk: (i + 1) * self._rows_per_chunk],
                string_max_len=self.string_max_len)
            # Let the gc collect the intermediate serialized chunk as early as possible
            chunk = chunk.tostring() if chunk is not None and get_bytes else chunk
            yield chunk, dtype

    def serialize(self):
        return self._serializer.serialize(self.original_df, self.string_max_len)
