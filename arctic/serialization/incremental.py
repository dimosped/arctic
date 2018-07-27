import abc
import logging

import numpy as np
import pandas as pd
from six.moves import xrange

from arctic.serialization.numpy_records import PandasSerializer
from ..exceptions import ArcticSerializationException
from ..store._ndarray_store import MAX_DOCUMENT_SIZE, _CHUNK_SIZE


ABC = abc.ABCMeta('ABC', (object,), {})
log = logging.getLogger(__name__)


class LazyIncrementalSerializer(ABC):
    def __init__(self, serializer, input_data, chunk_size=_CHUNK_SIZE):
        if chunk_size < 1:
            raise ArcticSerializationException("LazyIncrementalSerializer can't be initialized "
                                               "with chunk_size < 1 ({})".format(chunk_size))
        if not serializer:
            raise ArcticSerializationException("LazyIncrementalSerializer can't be initialized "
                                               "with a None serializer object")
        self.input_data = input_data
        self.chunk_size = chunk_size
        self._serializer = serializer
        self._initialized = False

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractproperty
    def generator(self):
        pass

    @abc.abstractproperty
    def generator_bytes(self):
        pass

    @abc.abstractproperty
    def serialize(self):
        pass


class IncrementalPandasToRecArraySerializer(LazyIncrementalSerializer):
    def __init__(self, serializer, input_data, chunk_size=_CHUNK_SIZE, string_max_len=None):
        super(IncrementalPandasToRecArraySerializer, self).__init__(serializer, input_data, chunk_size)
        if not isinstance(serializer, PandasSerializer):
            raise ArcticSerializationException("IncrementalPandasToRecArraySerializer requires a serializer of "
                                               "type PandasSerializer.")
        if not isinstance(input_data, (pd.DataFrame, pd.Series)):
            raise ArcticSerializationException("IncrementalPandasToRecArraySerializer requires a pandas DataFrame or "
                                               "Series as data source input.")
        if string_max_len and string_max_len < 1:
            raise ArcticSerializationException("IncrementalPandasToRecArraySerializer can't be initialized "
                                               "with string_max_len < 1 ({})".format(string_max_len))
        self.string_max_len = string_max_len
        # The state which needs to be lazily initialized
        self._dtype = None
        self._rows_per_chunk = 0
        self._total_chunks = 0
        self._has_string_object = False

    def _dtype_column_max_len_string(self, input_ndtype, fname):
        if input_ndtype.type not in (np.string_, np.unicode_):
            return input_ndtype, False
        type_sym = 'S' if input_ndtype.type == np.string_ else 'U'
        max_str_len = len(max(self.input_data[fname].astype(type_sym), key=len))
        str_field_dtype = np.dtype('{}{:d}'.format(type_sym, max_str_len)) if max_str_len > 0 else input_ndtype
        return str_field_dtype, True
    
    def _get_dtype(self):
        # Serializer is being called only if can_convert_to_records_without_objects() has passed,
        # which means that the resulting recarray does not contain objects but (only numpy types, string, or unicode)
        # TODO: We shouldn't fully serialize once with can_convert_to_records_without_objects() and then re-serialize.
        #       Instead add a faster implementation for can_convert_to_records_without_objects()
        #       to avoid double serialization when we have columns with (some) string values.

        # Serialize the first row to obtain info about row size in bytes (cache first row)
        # Also raise an Exception early, if data are not serializable
        first_chunk, dtype = self._serializer.serialize(self.input_data[0:1] if len(self) > 0 else self.input_data,
                                                        string_max_len=self.string_max_len)
        # This is the common case, where first row's dtype represents well the whole dataframe's dtype
        if dtype is None or len(self.input_data) == 0 or all(self.input_data.dtypes != object):
            return first_chunk, dtype, False

        # Reaching here means we have at least one column of type object
        # To correctly serialize incrementally, we need to know the final dtype (type and fixed length),
        # using length-conversion information from all values of the object columns
        dtype_arr = []
        has_string_object = False
        for field_name in dtype.names:  # include all column names, along with the expanded multi-index
            field_dtype = dtype[field_name]
            if field_name not in self.input_data or self.input_data.dtypes[field_name].hasobject:
                # if column is an expanded multi index or doesn't contain objects, the serialized 1st row dtype is safe
                field_dtype, has_string_object = self._dtype_column_max_len_string(field_dtype, field_name)
            dtype_arr.append((field_name, field_dtype))
        return first_chunk, np.dtype(dtype_arr), has_string_object

    def _lazy_init(self):
        if self._initialized:
            return
        
        # Get the dtype of the serialized array (takes into account object types, converted to fixed length strings)
        first_chunk, dtype, has_string_object = self._get_dtype()

        # Compute the number of rows which can fit in a chunk
        rows_per_chunk = 0
        if len(self) > 0 and self.chunk_size > 1:
            rows_per_chunk = IncrementalPandasToRecArraySerializer._calculate_rows_per_chunk(self.chunk_size, first_chunk)

        # Initialize object's state
        self._dtype = dtype
        self._has_string_object = has_string_object
        self._rows_per_chunk = rows_per_chunk
        self._total_chunks = int(np.ceil(float(len(self)) / self._rows_per_chunk)) if rows_per_chunk > 0 else 0
        self._initialized = True

    @staticmethod
    def _calculate_rows_per_chunk(max_chunk_size, chunk):
        sze = int(chunk.dtype.itemsize * np.prod(chunk.shape[1:]))
        sze = sze if sze < max_chunk_size else max_chunk_size
        rows_per_chunk = int(max_chunk_size / sze)
        if rows_per_chunk < 1:
            # If a row size is larger than chunk_size, use the maximum document size
            logging.warning('Chunk size of {} is too small to fit a row ({}). '
                            'Using maximum document size.'.format(max_chunk_size, MAX_DOCUMENT_SIZE))
            # For huge rows, fall-back to using a very large document size, less than max-allowed by MongoDB
            rows_per_chunk = int(MAX_DOCUMENT_SIZE / sze)
            if rows_per_chunk < 1:
                raise ArcticSerializationException("Serialization failed to split data into max sized chunks.")
        return rows_per_chunk

    def __len__(self):
        return len(self.input_data)

    @property
    def shape(self):
        return self.input_data.shape

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

        # Perform serialization for each chunk
        for i in xrange(self._total_chunks):
            chunk, _ = self._serializer.serialize(
                self.input_data[i * self._rows_per_chunk: (i + 1) * self._rows_per_chunk],
                string_max_len=self.string_max_len,
                forced_dtype=self.dtype if self._has_string_object else None)
            # Let the gc collect the intermediate serialized chunk as early as possible
            chunk = chunk.tostring() if chunk is not None and get_bytes else chunk
            yield chunk, self.dtype

    def serialize(self):
        return self._serializer.serialize(self.input_data, self.string_max_len)
