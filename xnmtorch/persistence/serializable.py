import inspect
import logging
from enum import Enum, auto
from functools import singledispatch, wraps
from typing import Any

import yaml

from xnmtorch.persistence.path import Path, PathError

logger = logging.getLogger("deserialize")


def _make_yaml_loader(cls):
    def from_yaml(loader, node):
        yaml_args = loader.construct_mapping(node)
        return UninitializedYamlObject(cls, yaml_args)

    return from_yaml


def _make_yaml_representer(yaml_tag):
    def to_yaml(dumper, data):
        return dumper.represent_mapping(yaml_tag, data._yaml_args)

    return to_yaml


class DeserializeError(RuntimeError):
    pass


def _serializable_init(cls, __init__):
    @wraps(__init__)
    def wrapper(self, *args, **kwargs):
        all_params = dict(kwargs)
        init_params = inspect.signature(__init__).parameters
        if len(args) > 0:
            param_names = [p.name for p in list(init_params.values())]
            assert param_names[0] == "self"
            param_names = param_names[1:]
            for name, arg in zip(param_names, args):
                all_params[name] = arg

        defaults = set()
        for param in init_params.values():
            if param.name != "self" and param.default != inspect.Parameter.empty \
                    and param.name not in all_params:
                all_params[param.name] = param.default
                defaults.add(param.name)

        for key, arg in list(all_params.items()):
            if isinstance(arg, Ref):
                if not arg.required:
                    all_params[key] = arg.default
                else:
                    if key in defaults:
                        raise ValueError(f"Required argument '{key}' of {type(self).__name__}.__init__() "
                                         f"was not specified, and {arg} could not be resolved")
                    else:
                        raise ValueError(f"Cannot pass a reference as argument; received {all_params[key]} "
                                         f"in {type(self).__name__}.__init__()")

        # for bare() default arguments
        for key, arg in list(all_params.items()):
            if isinstance(arg, UninitializedYamlObject):
                logger.debug(f"Initializing bare {arg.cls.__name__} in {cls.__name__}")
                initialized = arg.initialize()
                all_params[key] = initialized

        self._yaml_args = all_params
        __init__(self, **all_params)

    return wrapper


class Serializable:
    __slots__ = ("_yaml_args",)

    def __init_subclass__(cls, yaml_tag=None, **kwargs):
        super().__init_subclass__()
        if yaml_tag is None:
            yaml_tag = f"!{cls.__name__}"

        init_params = inspect.signature(cls.__init__).parameters
        for param in init_params.values():
            if param.default != inspect.Parameter.empty and isinstance(param.default, Serializable):
                logger.warning(f"{cls.__name__}.__init__ parameter {param.name} default is Serializable, "
                               f"this is not recommended, use bare({param.default.__class__.__name__}) instead."
                               f" Doing so will allow parameter sharing.")

        cls.__init__ = _serializable_init(cls, cls.__init__)

        yaml.FullLoader.add_constructor(yaml_tag, _make_yaml_loader(cls))
        yaml.Dumper.add_representer(cls, _make_yaml_representer(yaml_tag))

    shared_params = []

    @staticmethod
    def load(stream):
        return yaml.full_load(stream).initialize()

    def dump(self):
        return yaml.dump(self)

    def clone(self):
        """
        Construct a new object of the same class initialized with the same parameters.
        Note that this does not clone object state.
        """
        return self.__class__(**self._yaml_args)

    def save_processed_arg(self, key: str, val: Any):
        """
        Save a new value for an init argument (call from within ``__init__()``).

        Normally, the serialization mechanism makes sure that the same arguments are passed when creating the class
        initially based on a config file, and when loading it from a saved model. This method can be called from inside
        ``__init__()`` to save a new value that will be passed when loading the saved model. This can be useful when one
        doesn't want to recompute something every time (like a vocab) or when something has been passed via implicit
        referencing which might yield inconsistent result when loading the model to assemble a new model of different
        structure.

        Args:
          key: name of property, must match an argument of ``__init__()``
          val: new value; a :class:`Serializable` or basic Python type or list or dict of these
        """
        if key not in inspect.signature(self.__init__).parameters:
            raise ValueError(f"{key} is not an init argument of {self}")
        self._yaml_args[key] = val


def add_alias(yaml_tag, cls):
    assert issubclass(cls, Serializable)

    yaml.FullLoader.add_constructor(yaml_tag, _make_yaml_loader(cls))
    yaml.Dumper.add_representer(cls, _make_yaml_representer(yaml_tag))


class Ref(metaclass=yaml.YAMLObjectMetaclass):
    yaml_tag = "!Ref"
    yaml_loader = yaml.FullLoader
    yaml_dumper = yaml.Dumper

    __slots__ = ("_path", "_default")

    NO_DEFAULT = 1928437192847

    def __init__(self, path, default=NO_DEFAULT):
        self._path = path
        self._default = default

    @property
    def required(self):
        return self.default == Ref.NO_DEFAULT

    @property
    def default(self):
        return getattr(self, "_default", Ref.NO_DEFAULT)

    @property
    def path(self):
        return Path(self._path)

    def __str__(self):
        default_str = f", default={self.default}" if not self.required else ""

        return f"Ref(path={self.path}{default_str})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        h = hash(self.path)
        if not self.required:
            try:
                h |= hash(self.default)
            except TypeError as e:
                raise TypeError("Unhashable Ref") from e
        return h

    def __eq__(self, other):
        if not isinstance(other, Ref):
            return False
        return self.path == other.path and self.default == other.default

    @classmethod
    def from_yaml(cls, loader, node):
        args = loader.construct_mapping(node)
        return cls(**args)

    @classmethod
    def to_yaml(cls, dumper, data):
        mapping = {"path": data._path}
        if not data.required:
            mapping["default"] = data._default
        return dumper.represent_mapping(cls.yaml_tag, mapping)


def bare(cls, **kwargs):
    assert issubclass(cls, Serializable)
    return UninitializedYamlObject(cls, kwargs)


class UninitializedYamlObject:
    def __init__(self, cls, yaml_args):
        assert issubclass(cls, Serializable)
        self.cls = cls
        self.yaml_args = yaml_args
        self.obj = None

    def initialize(self):
        """
        Initialize this object and all children.
        This object is the root of the reference tree
        :return: Serializable
        """
        self._resolve_bare_and_ref_default_args()
        self._create_referenced_default_args()
        self._share_init_params_top_down()
        self._init_components_bottom_up()
        return self.obj

    @property
    def init_arg_defaults(self):
        return inspect.signature(self.cls.__init__).parameters

    def _resolve_bare_and_ref_default_args(self, prefix=Path()):
        """
        If an argument is not given in the YAML file and is set to a Ref or a bare object by default
        (e.g. dropout=Ref("exp_global.dropout")), copy this Ref into the yaml_args
        """
        for path, node in _traverse_tree(self):
            if isinstance(node, UninitializedYamlObject):
                init_args_defaults = node.init_arg_defaults
                for expected_arg in init_args_defaults:
                    if expected_arg not in node.yaml_args:
                        arg_default = init_args_defaults[expected_arg].default
                        abs_path = prefix.add_path(path).append(expected_arg)
                        if isinstance(arg_default, Ref):
                            logger.debug(f"Adding default Ref {abs_path} -> {arg_default.path}")
                            node.yaml_args[expected_arg] = arg_default
                        elif isinstance(arg_default, UninitializedYamlObject):
                            logger.debug(f"Adding default bare at {abs_path}")
                            arg_default._resolve_bare_and_ref_default_args(prefix=abs_path)
                            node.yaml_args[expected_arg] = arg_default

    def _create_referenced_default_args(self):
        """
        If there is a Ref to an argument that is not explicitly given,
        use the default value for that argument
        """
        for path, node in _traverse_tree(self):
            if isinstance(node, Ref):
                referenced_path = node.path
                give_up = False
                for ancestor in sorted(referenced_path.ancestors(), key=lambda x: len(x)):
                    try:
                        # If the path exists, do nothing
                        _get_descendant(self, ancestor)
                    except PathError:
                        try:
                            # If it doesn't, get the value from the default args
                            ancestor_parent = _get_descendant(self, ancestor.parent())
                        except PathError:
                            continue
                        if isinstance(ancestor_parent, UninitializedYamlObject):
                            init_args_defaults = ancestor_parent.init_arg_defaults
                            if ancestor[-1] in init_args_defaults:
                                referenced_arg_default = init_args_defaults[ancestor[-1]].default
                            else:
                                referenced_arg_default = inspect.Parameter.empty
                            if referenced_arg_default != inspect.Parameter.empty:
                                logger.debug(f"Adding Ref to implicit argument {path} -> {referenced_path}")
                                # make the default arg an implicit arg so the Ref can be resolved
                                _set_descendant(self, ancestor, referenced_arg_default)
                        else:
                            give_up = True
                    if give_up:
                        break

    def _share_init_params_top_down(self):
        """
        Share params as requested by Serializable.share_params
        """
        abs_shared_param_sets = []
        for path, node in _traverse_tree(self):
            if isinstance(node, UninitializedYamlObject):
                for shared_param_set in node.cls.shared_params:
                    shared_param_set = {Path(p) if isinstance(p, str) else p for p in shared_param_set}
                    abs_shared_param_set = {p.get_absolute(path) for p in shared_param_set}
                    added = False
                    for prev_set in abs_shared_param_sets:
                        if prev_set & abs_shared_param_set:
                            prev_set |= abs_shared_param_set
                            added = True
                            break
                    if not added:
                        abs_shared_param_sets.append(abs_shared_param_set)

        for shared_param_set in abs_shared_param_sets:
            shared_val_choices = set()
            for shared_param_path in shared_param_set:
                try:
                    new_shared_val = _get_descendant(self, shared_param_path)
                except PathError:
                    continue

                for _, child_of_shared_param in _traverse_tree(new_shared_val, include_root=False):
                    if isinstance(child_of_shared_param, UninitializedYamlObject):
                        logger.warning(f"{shared_param_path} shared params {shared_param_set} contains Serializable"
                                       f"sub-object {child_of_shared_param} which does not support parameter sharing")
                shared_val_choices.add(new_shared_val)
            if len(shared_val_choices) == 0:
                # try and go by defaults
                for shared_param_path in shared_param_set:
                    try:
                        parent = _get_descendant(self, shared_param_path.parent())
                    except PathError:
                        continue
                    if not isinstance(parent, UninitializedYamlObject):
                        continue
                    init_args_defaults = parent.init_arg_defaults
                    param_name = shared_param_path[-1]
                    if param_name in init_args_defaults and \
                            init_args_defaults[param_name].default != inspect.Parameter.empty:
                        shared_val_choices.add(init_args_defaults[param_name].default)
                    else:
                        continue

            if len(shared_val_choices) == 0:
                logger.warning(f"No param choices for {shared_param_set}")
            elif len(shared_val_choices) > 1:
                raise DeserializeError(f"inconsistent shared params for {shared_param_set}: "
                                       f"{shared_val_choices}; Ignoring these shared parameters.")
            else:
                logger.debug(f"Sharing parameters {shared_param_set} = {next(iter(shared_val_choices))}")
                for shared_param_path in shared_param_set:
                    try:
                        descendant = _get_descendant(self, shared_param_path.parent())
                    except PathError:
                        # can happen when the shared path contained a reference,
                        # which we don't follow to avoid unwanted effects
                        continue
                    if descendant is None:
                        continue
                    elif not isinstance(descendant, UninitializedYamlObject):
                        raise ValueError(f"Error when trying to share attribute {shared_param_path[-1]} "
                                         f"of {type(descendant).__name__}")
                    elif shared_param_path[-1] in descendant.init_arg_defaults:
                        _set_descendant(self, shared_param_path, list(shared_val_choices)[0])

    def _init_components_bottom_up(self):
        cache = dict()
        for path, node in _traverse_tree_deep_once(self, self, _TraversalOrder.ROOT_LAST):
            if isinstance(node, Ref):
                cache_size = len(cache)
                initialized_component = self._resolve_ref(node, cache)
                logger.debug(f"Resolved Ref {path} -> {node.path}")
                if len(cache) == cache_size:
                    logger.debug(f"For {path}: Reusing {_obj_to_str(initialized_component)}")
            elif isinstance(node, UninitializedYamlObject):
                for name, _ in _name_children(node):
                    initialized_child = cache[path.append(name)]
                    node.yaml_args[name] = initialized_child

                initialized_component = node._init_component()
                logger.debug(f"In {path}: Initialized {_obj_to_str(initialized_component)}")
            elif isinstance(node, list):
                initialized_component = [cache[path.append(str(i))] for i in range(len(node))]
            elif isinstance(node, dict):
                initialized_component = {key: cache[path.append(key)] for key in node.keys()}
            else:
                initialized_component = node
            cache[path] = initialized_component

    def _resolve_ref(self, node: Ref, cache: dict):
        resolved_path = node.path
        if resolved_path in cache:
            return cache[resolved_path]
        else:
            try:
                initialized_component = _get_descendant(self, resolved_path)
                if isinstance(initialized_component, UninitializedYamlObject):
                    initialized_component = initialized_component._init_component()
            except PathError as e:
                if node.required:
                    # initialized_component = None
                    raise ReferenceError(str(resolved_path)) from e
                else:
                    initialized_component = node.default
            cache[resolved_path] = initialized_component
            return initialized_component

    def _init_component(self):
        if self.obj is not None:
            return self.obj
        # check types
        if any(isinstance(param, UninitializedYamlObject) for param in self.yaml_args.values()):
            cls_name = next(p.cls.__name__ for p in self.yaml_args.values() if isinstance(p, UninitializedYamlObject))
            raise DeserializeError(f"In deserialization of {self.cls.__name__}: "
                                   f"found uninitialized yaml object of type {cls_name}")
        try:
            obj = self.cls(**self.yaml_args)
        except Exception as e:
            raise DeserializeError(f"Deserialization of {self.cls.__name__} failed") from e
        self.obj = obj
        return obj

    def __str__(self):
        return f"UninitializedYamlObject<{self.cls.__name__}>@{id(self)}({self.yaml_args.keys()})"


def _obj_to_str(obj):
    if isinstance(obj, (int, str, float, bool, UninitializedYamlObject)):
        return str(obj)
    else:
        return obj.__class__.__name__


class _TraversalOrder(Enum):
    ROOT_FIRST = auto()
    ROOT_LAST = auto()


def _traverse_tree(node, traversal_order=_TraversalOrder.ROOT_FIRST, path_to_node=Path(), include_root=True):
    """
    For each node in the tree, yield a (path, node) tuple
    """
    if include_root and traversal_order == _TraversalOrder.ROOT_FIRST:
        yield path_to_node, node
    for child_name, child in _name_children(node):
        yield from _traverse_tree(child, traversal_order, path_to_node.append(child_name))
    if include_root and traversal_order == _TraversalOrder.ROOT_LAST:
        yield path_to_node, node


@singledispatch
def _name_children(node):
    return []


@_name_children.register(UninitializedYamlObject)
def _name_children_serializable(node: UninitializedYamlObject):
    """
    Returns the specified arguments in the order they appear in the corresponding ``__init__()``
    """
    init_args = list(node.init_arg_defaults.keys())

    # if include_reserved: init_args += [n for n in _reserved_arg_names if not n in init_args]
    ret = []
    for name in init_args:
        if name in node.yaml_args:
            val = node.yaml_args[name]
            ret.append((name, val))
    return ret


@_name_children.register(dict)
def _name_children_dict(node: dict):
    return node.items()


@_name_children.register(list)
def _name_children_list(node: list):
    return [(str(n), l) for n, l in enumerate(node)]


@_name_children.register(tuple)
def _name_children_tuple(node):
    raise ValueError(f"Tuples are not serializable, use a list instead. Found this tuple: {node}.")


@singledispatch
def _get_child(node, name):
    if not hasattr(node, name):
        raise PathError(f"{node} has no child named {name}")
    return getattr(node, name)


@_get_child.register(list)
def _get_child_list(node, name):
    try:
        name = int(name)
    except ValueError:
        raise PathError(f"{node} has no child named {name} (integer expected)")
    if not 0 <= name < len(node):
        raise PathError(f"{node} has no child named {name} (index error)")
    return node[int(name)]


@_get_child.register(dict)
def _get_child_dict(node, name):
    if name not in node.keys():
        raise PathError(f"{node} has no child named {name} (key error)")
    return node[name]


@_get_child.register(UninitializedYamlObject)
def _get_child_serializable(node: UninitializedYamlObject, name):
    if name not in node.yaml_args:
        raise PathError(f"{node} has no child named {name}")
    return node.yaml_args[name]


@singledispatch
def _set_child(node, name, val):
    pass


@_set_child.register(UninitializedYamlObject)
def _set_child_serializable(node: UninitializedYamlObject, name, val):
    node.yaml_args[name] = val


@_set_child.register(list)
def _set_child_list(node, name, val):
    if name == "append":
        name = len(node)
    try:
        name = int(name)
    except ValueError:
        raise PathError(f"{node} has no child named {name} (integer expected)")
    if not 0 <= name < len(node) + 1:
        raise PathError(f"{node} has no child named {name} (index error)")
    if name == len(node):
        node.append(val)
    else:
        node[int(name)] = val


@_set_child.register(dict)
def _set_child_dict(node, name, val):
    node[name] = val


def _get_descendant(node: UninitializedYamlObject, path: Path, redirect=False):
    if len(path) == 0:
        return node
    elif redirect and isinstance(node, Ref):
        node_path = node.path
        return Ref(node_path.add_path(path), default=node.default)
    else:
        return _get_descendant(_get_child(node, path[0]), path.descend_one(), redirect=redirect)


def _set_descendant(root, path, val):
    if len(path) == 0:
        raise ValueError("path was empty")
    elif len(path) == 1:
        _set_child(root, path[0], val)
    else:
        _set_descendant(_get_child(root, path[0]), path.descend_one(), val)


def _traverse_tree_deep(root, cur_node, traversal_order=_TraversalOrder.ROOT_FIRST, path_to_node=Path(),
                        past_visits=frozenset()):
    """
    Traverse the tree and descend into references. The returned path is that of the resolved reference.

    Args:
      root (UninitializedYamlObject):
      cur_node (UninitializedYamlObject):
      traversal_order (_TraversalOrder):
      path_to_node (Path):
      past_visits (set):
    """

    # prevent infinite recursion:
    cur_call_sig = (id(root), id(cur_node), path_to_node)
    if cur_call_sig in past_visits: return
    past_visits = set(past_visits)
    past_visits.add(cur_call_sig)

    if traversal_order == _TraversalOrder.ROOT_FIRST:
        yield path_to_node, cur_node
    if isinstance(cur_node, Ref):
        resolved_path = cur_node.path
        try:
            yield from _traverse_tree_deep(root, _get_descendant(root, resolved_path), traversal_order, resolved_path,
                                           past_visits=past_visits)
        except PathError:
            pass
    else:
        for child_name, child in _name_children(cur_node):
            yield from _traverse_tree_deep(root, child, traversal_order, path_to_node.append(child_name),
                                           past_visits=past_visits)
    if traversal_order == _TraversalOrder.ROOT_LAST:
        yield path_to_node, cur_node


def _traverse_tree_deep_once(root, cur_node, traversal_order=_TraversalOrder.ROOT_FIRST, path_to_node=Path()):
    """
    Calls _traverse_tree_deep, but skips over nodes that have been visited before
    (can happen because we're descending into references).
    """
    yielded_paths = set()
    for path, node in _traverse_tree_deep(root, cur_node, traversal_order, path_to_node):
        if not (path.ancestors() & yielded_paths):
            yielded_paths.add(path)
            yield (path, node)
