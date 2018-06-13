import logging
from threading import RLock
import time

import pymongo
import uuid
from concurrent.futures import ThreadPoolExecutor, wait, as_completed, FIRST_EXCEPTION

from arctic.exceptions import ArcticException
from ..decorators import mongo_retry

MONGO_ASYNC_NTHREADS = 2
BULK_WRITE_BATCH_SIZE = 10
_MONGO_RETRY_FNAME = mongo_retry.func_name
_SINGLETON_LOCK = RLock()


def _mongo_exec(request):
    request.start_time = time.time()
    if request.with_retry:
        result = mongo_retry(request.fun)(*request.args, **request.kwargs)
    else:
        result = request.fun(*request.args, **request.kwargs)
    if isinstance(result, pymongo.cursor.Cursor):
        request.data = list(result)
        result = request.data
    request.end_time = time.time()
    return result


class MongoRequest(object):
    def __init__(self, fun, with_retry, *args, **kwargs):
        self.id = uuid.uuid4()
        self.fun = fun
        self.args = args
        self.kwargs = kwargs
        self.with_retry = with_retry
        # Request's state
        self.future = None
        self.data = None
        self._executing_batch = None
        # Timekeeping
        self.start_time = None
        self.end_time = None
        self.create_time = time.time()


class MongoAsync(object):
    __instance = None

    @staticmethod
    def get_instance():
        # Lazy init
        with _SINGLETON_LOCK:
            if MongoAsync.__instance is None:
                MongoAsync.__instance = MongoAsync()
        return MongoAsync.__instance

    def __init__(self):
        with _SINGLETON_LOCK:
            if MongoAsync.__instance is not None:
                raise ArcticException("MongoAsync is a singleton, can't create a new instance")
            self._lock = RLock()
            self._bulk_write_batches = {}
            self._pool = ThreadPoolExecutor(max_workers=MONGO_ASYNC_NTHREADS, thread_name_prefix='MongoAsyncWorker')

    def __reduce__(self):
        return "MONGO_ASYNC"
    
    def submit_request(self, fun, *args, **kwargs):
        with_mongo_retry = _MONGO_RETRY_FNAME in kwargs
        if with_mongo_retry:
            del kwargs[_MONGO_RETRY_FNAME]
        request = MongoRequest(fun, with_mongo_retry, *args, **kwargs)
        request.future = self._pool.submit(_mongo_exec, request)
        return request

    @staticmethod
    def _join_batch(request):
        # Always call from a synchronized caller
        if request is not None and request.future is not None:
            wait((request.future,), return_when=FIRST_EXCEPTION)
            # Force-raise any exceptions
            request.future.result()
            request.future = None
            request.data = None
            del request.args[0][:]

    def _start_new_batch(self, request):
        # If next batch is ready for dispatch, wait for the previous batch to finish first
        self._join_batch(request._executing_batch)
        self._join_batch(request)  # not necessary, just guard if caller is sloppy

        # Update timestamps
        request.start_time = request._executing_batch.start_time if request.start_time is None else request.start_time
        request.end_time = request._executing_batch.end_time

        # Stop if we have no more operations to execute
        if len(request.args[0]) < 1:
            return

        # Transfer the new ops to the executing batch request
        # del request._executing_batch.args[0][:]
        request._executing_batch.args[0].extend(request.args[0])

        # Free up the waiting batch operations
        del request.args[0][:]

        # Execute the batch
        request._executing_batch.future = self._pool.submit(_mongo_exec, request._executing_batch)

    def submit_bulk_write(self, collection, op,
                          ordered=False, with_mongo_retry=False, request=None, batch_size=BULK_WRITE_BATCH_SIZE):
        if op is None:
            return request

        # Init the request objects if necessary
        if request is None:
            request = MongoRequest(collection.bulk_write, with_mongo_retry,
                                   *([],), **{'ordered': ordered})
        if request._executing_batch is None:
            request._executing_batch = MongoRequest(collection.bulk_write, with_mongo_retry,
                                                    *([],), **{'ordered': ordered})

        with self._lock:
            request.args[0].append(op)

            if len(request.args[0]) >= batch_size:
                # Start the new batch
                self._start_new_batch(request)

        return request

    def join_bulk_write(self, request):
        if request is None:
            return

        with self._lock:
            # Run any outsanding operations
            self._start_new_batch(request)
            self._join_batch(request._executing_batch)
            request._executing_batch = None

    def reset(self, wait=True, pool_size=MONGO_ASYNC_NTHREADS):
        self._pool.shutdown(wait=wait)
        pool_size = max(pool_size, 1)
        self._pool = ThreadPoolExecutor(pool_size)

