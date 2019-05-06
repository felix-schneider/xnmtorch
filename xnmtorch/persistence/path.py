from typing import Set


class Path:
    """
    A relative or absolute path in the component hierarchy.

    Paths are immutable: Operations that change the path always return a new Path object.

    Args:
      path_str: path string, with period ``.`` as separator. If prefixed by ``.``, marks a relative path, otherwise
                absolute.
    """

    __slots__ = ("path_str",)

    def __init__(self, path_str: str = "") -> None:
        if (len(path_str) > 1 and path_str[-1] == "." and path_str[-2] != ".") \
                or ".." in path_str.strip("."):
            # TODO: Begin and end with any number of .?
            raise ValueError(f"'{path_str}' is not a valid path string")
        self.path_str = path_str

    @property
    def relative(self) -> bool:
        return self.path_str.startswith(".")

    def append(self, link: str) -> 'Path':
        """
        Return a new path by appending a link.

        Args:
          link: link to append

        Returns: new path

        """
        if not link or "." in link:
            raise ValueError(f"'{link}' is not a valid link")
        if len(self.path_str.strip(".")) == 0:
            return Path(f"{self.path_str}{link}")
        else:
            return Path(f"{self.path_str}.{link}")

    def add_path(self, path_to_add: 'Path') -> 'Path':
        """
        Concatenates a path

        Args:
          path_to_add: path to concatenate

        Returns: concatenated path

        """
        if path_to_add.relative:
            raise NotImplementedError("add_path() is not implemented for relative paths.")

        if len(self.path_str.strip(".")) == 0 or len(path_to_add.path_str) == 0:
            return Path(f"{self.path_str}{path_to_add.path_str}")
        else:
            return Path(f"{self.path_str}.{path_to_add.path_str}")

    def get_absolute(self, rel_to: 'Path') -> 'Path':
        if rel_to.relative:
            raise ValueError("rel_to must be an absolute path!")
        if self.relative:
            num_up = len(self.path_str) - len(self.path_str.strip(".")) - 1
            for _ in range(num_up):
                rel_to = rel_to.parent()
            s = self.path_str.strip(".")
            if len(s) > 0:
                for link in s.split("."):
                    rel_to = rel_to.append(link)
            return rel_to
        else:
            return self

    def descend_one(self) -> 'Path':
        if self.relative or len(self) == 0:
            raise ValueError(f"Can't call descend_one() on path {self.path_str}")
        return Path(".".join(self.path_str.split(".")[1:]))

    def ancestors(self) -> Set['Path']:
        a = self
        ret = {a}
        while len(a.path_str.strip(".")) > 0:
            a = a.parent()
            ret.add(a)
        return ret

    def parent(self) -> 'Path':
        if len(self.path_str.strip(".")) == 0:
            raise ValueError(f"Path '{self.path_str}' has no parent")
        else:
            spl = self.path_str.split(".")[:-1]
            if '.'.join(spl) == "" and self.path_str.startswith("."):
                return Path(".")
            else:
                return Path(".".join(spl))

    def __str__(self):
        return self.path_str

    def __repr__(self):
        return self.path_str

    def __len__(self):
        if self.relative:
            raise ValueError(f"Can't call __len__() on path {self.path_str}")
        if len(self.path_str) == 0:
            return 0
        return len(self.path_str.split("."))

    def __getitem__(self, key):
        if self.relative:
            raise ValueError(f"Can't call __getitem__() on path {self.path_str}")
        if isinstance(key, slice):
            _, _, step = key.indices(len(self))
            if step is not None and step != 1: raise ValueError(f"step must be 1, found {step}")
            return Path(".".join(self.path_str.split(".")[key]))
        else:
            return self.path_str.split(".")[key]

    def __hash__(self):
        return hash(self.path_str)

    def __eq__(self, other):
        if isinstance(other, Path):
            return self.path_str == other.path_str
        else:
            return False


class PathError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
