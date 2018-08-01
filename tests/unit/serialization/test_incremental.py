import pytest
import numpy as np
import pandas as pd

from arctic.exceptions import ArcticSerializationException
from arctic.serialization.incremental import IncrementalPandasToRecArraySerializer
from arctic.serialization.numpy_records import DataFrameSerializer
from tests.integration.chunkstore.test_utils import create_test_data

from tests.util import get_large_ts

NON_HOMOGENEOUS_DTYPE_PATCH_SIZE_ROWS = 50
_TEST_DATA = None

df_serializer = DataFrameSerializer()


def _test_data():
    global _TEST_DATA
    if _TEST_DATA is None:
        onerow_ts = get_large_ts(1)
        small_ts = get_large_ts(10)
        medium_ts = get_large_ts(600)
        large_ts = get_large_ts(1800)
        empty_ts = pd.DataFrame()
        empty_index = create_test_data(size=0, cols=10, index=True, multiindex=False, random_data=True, random_ids=True)

        with_some_objects_ts = medium_ts.copy(deep=True)
        with_some_objects_ts.iloc[0:NON_HOMOGENEOUS_DTYPE_PATCH_SIZE_ROWS, 0] = None
        with_some_objects_ts.iloc[0:NON_HOMOGENEOUS_DTYPE_PATCH_SIZE_ROWS, 1] = 'A string'
        large_with_some_objects = create_test_data(size=10000, cols=64, index=True, multiindex=False, random_data=True, random_ids=True, use_hours=True)
        large_with_some_objects.iloc[0:NON_HOMOGENEOUS_DTYPE_PATCH_SIZE_ROWS, 0] = None
        large_with_some_objects.iloc[0:NON_HOMOGENEOUS_DTYPE_PATCH_SIZE_ROWS, 1] = 'A string'

        with_string_ts = medium_ts.copy(deep=True)
        with_string_ts['str_col'] = 'abc'
        with_unicode_ts = medium_ts.copy(deep=True)
        with_unicode_ts['ustr_col'] = u'abc'

        with_some_none_ts = medium_ts.copy(deep=True)
        with_some_none_ts.iloc[10:10] = None
        with_some_none_ts.iloc[-10:-10] = np.nan
        with_some_none_ts = with_some_none_ts.replace({np.nan: None})

        # Multi-index data frames
        multiindex_ts = create_test_data(size=500, cols=10, index=True, multiindex=True, random_data=True,
                                         random_ids=True)
        empty_multiindex_ts = create_test_data(size=0, cols=10, index=True, multiindex=True, random_data=True,
                                               random_ids=True)
        large_multi_index = create_test_data(
            size=50000, cols=10, index=True, multiindex=True, random_data=True, random_ids=True, use_hours=True)

        # Multi-column data frames
        columns = pd.MultiIndex.from_product([["bar", "baz", "foo", "qux"], ["one", "two"]], names=["first", "second"])
        empty_multi_column_ts = pd.DataFrame([], columns=columns)

        columns = pd.MultiIndex.from_product([["bar", "baz", "foo", "qux"], ["one", "two"]], names=["first", "second"])
        multi_column_no_multiindex = pd.DataFrame(np.random.randn(2, 8), index=[0, 1], columns=columns)

        large_multi_column = pd.DataFrame(np.random.randn(100000, 8), index=range(100000), columns=columns)

        columns = pd.MultiIndex.from_product([[1, 2, 'a'], ['c', 5]])
        multi_column_int_levels = pd.DataFrame([[9, 2, 8, 1, 2, 3], [3, 4, 2, 9, 10, 11]], index=['x', 'y'], columns=columns)

        # Multi-index and multi-column data frames
        columns = pd.MultiIndex.from_product([["bar", "baz", "foo", "qux"], ["one", "two"]])
        index = pd.MultiIndex.from_product([["x", "y", "z"], ["a", "b"]])
        multi_column_and_multi_index = pd.DataFrame(np.random.randn(6, 8), index=index, columns=columns)

        # Nested n-dimensional
        def _new_np_nd_array(val):
            return np.rec.array([(val, ['A', 'BC'])],
                                dtype=[('index', '<M8[ns]'), ('values', 'S2', (2,))])
        n_dimensional_df = pd.DataFrame(
            {'a': [_new_np_nd_array(1356998400000000000), _new_np_nd_array(1356998400000000001)],
             'b': [_new_np_nd_array(1356998400000000002), _new_np_nd_array(1356998400000000003)]
             },
            index=(0, 1))

        _TEST_DATA = {
            'onerow': (onerow_ts, df_serializer.serialize(onerow_ts)),
            'small': (small_ts, df_serializer.serialize(small_ts)),
            'medium': (medium_ts, df_serializer.serialize(medium_ts)),
            'large': (large_ts, df_serializer.serialize(large_ts)),
            'empty': (empty_ts, df_serializer.serialize(empty_ts)),
            'empty_index': (empty_index, df_serializer.serialize(empty_index)),
            'with_some_objects': (with_some_objects_ts, df_serializer.serialize(with_some_objects_ts)),
            'large_with_some_objects': (large_with_some_objects, df_serializer.serialize(large_with_some_objects)),
            'with_string': (with_string_ts, df_serializer.serialize(with_string_ts)),
            'with_unicode': (with_unicode_ts, df_serializer.serialize(with_unicode_ts)),
            'with_some_none': (with_some_none_ts, df_serializer.serialize(with_some_none_ts)),
            'multiindex': (multiindex_ts, df_serializer.serialize(multiindex_ts)),
            'empty_multiindex': (empty_multiindex_ts, df_serializer.serialize(empty_multiindex_ts)),
            'large_multi_index': (large_multi_index, df_serializer.serialize(large_multi_index)),
            'empty_multicolumn': (empty_multi_column_ts, df_serializer.serialize(empty_multi_column_ts)),
            'multi_column_no_multiindex': (multi_column_no_multiindex, df_serializer.serialize(multi_column_no_multiindex)),
            'large_multi_column': (large_multi_column, df_serializer.serialize(large_multi_column)),
            'multi_column_int_levels': (multi_column_int_levels, df_serializer.serialize(multi_column_int_levels)),
            'multi_column_and_multi_index': (multi_column_and_multi_index, df_serializer.serialize(multi_column_and_multi_index)),
            'n_dimensional_df': (n_dimensional_df, Exception)
        }
    return _TEST_DATA


def test_incremental_bad_init():
    with pytest.raises(ArcticSerializationException):
        IncrementalPandasToRecArraySerializer(df_serializer, 'hello world')
    with pytest.raises(ArcticSerializationException):
        IncrementalPandasToRecArraySerializer(df_serializer, 1234)
    with pytest.raises(ArcticSerializationException):
        IncrementalPandasToRecArraySerializer(df_serializer, _test_data()['small'][0], chunk_size=0)
    with pytest.raises(ArcticSerializationException):
        IncrementalPandasToRecArraySerializer(df_serializer, _test_data()['small'][0], chunk_size=-1)
    with pytest.raises(ArcticSerializationException):
        IncrementalPandasToRecArraySerializer(df_serializer, _test_data()['small'][0], string_max_len=-1)


def test_none_df():
    with pytest.raises(ArcticSerializationException):
        incr_ser = IncrementalPandasToRecArraySerializer(df_serializer, None)
        incr_ser.serialize()


@pytest.mark.parametrize("input_df", _test_data().keys())
def test_serialize_pandas_to_recarray(input_df):
    df = _test_data()[input_df][0]
    expectation = _test_data()[input_df][1]

    incr_ser = IncrementalPandasToRecArraySerializer(df_serializer, df)
    if not isinstance(expectation, tuple) and issubclass(expectation, Exception):
        with pytest.raises(expectation):
            incr_ser.serialize()
    else:
        incr_ser_data, incr_ser_dtype = incr_ser.serialize()
        matching = expectation[0].tostring() == incr_ser_data.tostring()
        assert matching
        assert expectation[1] == incr_ser_dtype


@pytest.mark.parametrize("input_df", _test_data().keys())
def test_serialize_incremental_pandas_to_recarray(input_df):
    df = _test_data()[input_df][0]
    expectation = _test_data()[input_df][1]

    incr_ser = IncrementalPandasToRecArraySerializer(df_serializer, df)

    if not isinstance(expectation, tuple) and issubclass(expectation, Exception):
        with pytest.raises(expectation):
            incr_ser.serialize()
    else:
        chunk_bytes = [chunk_b for chunk_b, _ in incr_ser.generator_bytes]
        matching = expectation[0].tostring() == b''.join(chunk_bytes)
        assert matching
        assert expectation[1] == incr_ser.dtype


@pytest.mark.parametrize("input_df", _test_data().keys())
def test_serialize_incremental_chunk_size_pandas_to_recarray(input_df):
    df = _test_data()[input_df][0]
    expectation = _test_data()[input_df][1]

    if not isinstance(expectation, tuple) and issubclass(expectation, Exception):
        for div in (1, 4, 8):
            chunk_size = div * 8 * 1024 ** 2
            with pytest.raises(expectation):
                incr_ser = IncrementalPandasToRecArraySerializer(df_serializer, df, chunk_size=chunk_size)
                incr_ser.serialize()
        return

    for div in (1, 4, 8):
        chunk_size = div * 8 * 1024 ** 2
        if input_df is not None and len(expectation) > 0:
            row_size = int(expectation[0].dtype.itemsize)
            chunk_size = NON_HOMOGENEOUS_DTYPE_PATCH_SIZE_ROWS * row_size / div
        incr_ser = IncrementalPandasToRecArraySerializer(df_serializer, df, chunk_size=chunk_size)
        chunk_bytes = [chunk[0] for chunk in incr_ser.generator_bytes]
        matching = expectation[0].tostring() == b''.join(chunk_bytes)
        assert matching
        assert expectation[1] == incr_ser.dtype
