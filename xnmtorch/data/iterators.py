import logging
import math

from torchtext.data.utils import RandomShuffler

from xnmtorch.data.batch import Batch

logger = logging.getLogger("iterator")


class Iterator(object):
    """Defines an iterator that loads batches of data from a Dataset.

    Attributes:
        dataset: The Dataset object to load Examples from.
        batch_size: Batch size.
        batch_size_fn: Function of three arguments (new example to add, current
            count of examples in the batch, and current effective batch size)
            that returns the new effective batch size resulting from adding
            that example to a batch. This is useful for dynamic batching, where
            this function would add to the current effective batch size the
            number of tokens in the new example.
        sort_key: A key to use for sorting examples in order to batch together
            examples with similar lengths and minimize padding. The sort_key
            provided to the Iterator constructor overrides the sort_key
            attribute of the Dataset, or defers to it if None.
        repeat: Whether to repeat the iterator for multiple epochs. Default: False.
        shuffle: Whether to shuffle examples between epochs.
        sort: Whether to sort examples according to self.sort_key.
            Note that repeat, shuffle, and sort default to train, train, and
            (not train).
        sort_within_batch: Whether to sort (in descending order according to
            self.sort_key) within each batch. If None, defaults to self.sort.
            If self.sort is True and this is False, the batch is left in the
            original (ascending) sorted order.
        device (str or `torch.device`): A string or instance of `torch.device`
            specifying which device the Variables are going to be created on.
            If left as default, the tensors will be created on cpu. Default: None.
    """

    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=None,
                 repeat=False, shuffle=False, sort=False,
                 sort_within_batch=False):
        self.batch_size, self.dataset = batch_size, dataset
        self.batch_size_fn = batch_size_fn
        self.batch_indices = None
        self.batch_sizes = None
        self.iterations = 0
        self.repeat = repeat
        self.shuffle = shuffle
        self.sort = sort
        self.sort_within_batch = sort_within_batch

        if sort_key is None:
            self.sort_key = dataset.sort_key
        else:
            self.sort_key = sort_key

        if type(device) == int:
            logger.warning("The `device` argument should be set by using `torch.device`" +
                           " or passing a string as an argument. This behavior will be" +
                           " deprecated soon and currently defaults to cpu.")
            device = None
        self.device = device
        self.random_shuffler = RandomShuffler()

        # For state loading/saving only
        self._iterations_this_epoch = 0
        self._random_state_this_epoch = None
        self._restored_from_state = False

    @classmethod
    def splits(cls, datasets, batch_sizes=None, **kwargs):
        """Create Iterator objects for multiple splits of a dataset.

        Arguments:
            datasets: Tuple of Dataset objects corresponding to the splits. The
                first such object should be the train set.
            batch_sizes: Tuple of batch sizes to use for the different splits,
                or None to use the same batch_size for all splits.
            Remaining keyword arguments: Passed to the constructor of the
                iterator class being used.
        """
        if batch_sizes is None:
            batch_sizes = [kwargs.pop('batch_size')] * len(datasets)
        ret = []
        for i in range(len(datasets)):
            ret.append(cls(
                datasets[i], batch_size=batch_sizes[i], **kwargs))
        return tuple(ret)

    def data_indices(self):
        if self.sort:
            logger.info("Sorting dataset...")
            xs = sorted(range(len(self.dataset)), key=lambda i: self.sort_key(self.dataset[i]))
        elif self.shuffle:
            logger.info("Shuffling...")
            xs = self.random_shuffler(range(len(self.dataset)))
        else:
            xs = range(len(self.dataset))
        return xs

    def data(self):
        for i in self.data_indices():
            yield self.dataset[i]

    def init_epoch(self):
        """Set up the batch generator for a new epoch."""
        logger.info("Creating batches")

        if self._restored_from_state:
            self.random_shuffler.random_state = self._random_state_this_epoch
        else:
            self._random_state_this_epoch = self.random_shuffler.random_state

        self.create_batches()

        logger.info(f"Batching complete, {len(self.batch_indices)} batches")

        if self._restored_from_state:
            self._restored_from_state = False
        else:
            self._iterations_this_epoch = 0

        if not self.repeat:
            self.iterations = 0

    def create_batches(self):
        self.batch_indices = []
        self.batch_sizes = []
        for indices, size in batch_indices(self.data_indices(), self.batch_size, self.batch_size_fn):
            self.batch_indices.append(indices)
            self.batch_sizes.append(size)

    @property
    def epoch(self):
        return math.floor(self.iterations / len(self))

    def __len__(self):
        if self.batch_indices is not None:
            return len(self.batch_indices)
        elif self.batch_size_fn is None:
            return math.ceil(len(self.dataset) / self.batch_size)
        else:
            raise NotImplementedError

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, (minibatch_indices, minibatch_size) in enumerate(zip(self.batch_indices, self.batch_sizes)):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                minibatch = [self.dataset[ind] for ind in minibatch_indices]
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                batch = Batch(minibatch, self.dataset, self.device)
                batch.batch_size = minibatch_size
                yield batch
            if not self.repeat:
                return

    def state_dict(self):
        return {
            "iterations": self.iterations,
            "iterations_this_epoch": self._iterations_this_epoch,
            "random_state_this_epoch": self._random_state_this_epoch}

    def load_state_dict(self, state_dict):
        self.iterations = state_dict["iterations"]
        self._iterations_this_epoch = state_dict["iterations_this_epoch"]
        self._random_state_this_epoch = state_dict["random_state_this_epoch"]
        self._restored_from_state = True


class BatchShuffledIterator(Iterator):
    def data_indices(self):
        if self.sort:
            logger.info("Sorting dataset...")
            xs = sorted(range(len(self.dataset)), key=lambda i: self.sort_key(self.dataset[i]))
        else:
            xs = range(len(self.dataset))
        return xs

    def create_batches(self):
        super().create_batches()
        if self.shuffle:
            logger.info("Shuffling batches")
            permutation = self.random_shuffler(range(len(self.batch_indices)))
            self.batch_indices = [self.batch_indices[i] for i in permutation]
            self.batch_sizes = [self.batch_sizes[i] for i in permutation]


def batch_indices(data_indices, batch_size, batch_size_fn=None):
    if batch_size_fn is None:
        def batch_size_fn(new, count, sofar):
            return count + 1
    minibatch, size_so_far = [], 0
    for i, ind in enumerate(data_indices):
        size_with_item = batch_size_fn(ind, len(minibatch), size_so_far)
        if size_with_item <= batch_size:
            minibatch.append(ind)
            size_so_far = size_with_item
        else:
            yield minibatch, size_so_far
            minibatch, size_so_far = [ind], batch_size_fn(ind, 0, 0)
    if len(minibatch) > 0:
        yield minibatch, size_so_far
