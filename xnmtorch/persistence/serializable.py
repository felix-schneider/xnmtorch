import inspect
import logging
from enum import Enum, auto
from functools import singledispatch, wraps

import yaml

from xnmtorch.persistence.format_string import FormatString
from xnmtorch.persistence.path import Path, PathError


logger = logging.getLogger("persistence")


def _make_yaml_loader(cls):
    def from_yaml(loader, node):
        yaml_args = loader.construct_mapping(node)
        return UninitializedYamlObject(cls, yaml_args)
    return from_yaml


def _make_yaml_representer(yaml_tag):
    def to_yaml(dumper, data):
        return dumper.represent_mapping(yaml_tag, data._yaml_args)
    return to_yaml


def _serializable_init(__init__):
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

        __init__(self, **all_params)
        self._yaml_args = all_params
    return wrapper


class Serializable:
    __slots__ = ("_yaml_args",)

    def __init_subclass__(cls, yaml_tag=None, **kwargs):
        super().__init_subclass__()
        if yaml_tag is None:
            yaml_tag = f"!{cls.__name__}"

        cls.__init__ = _serializable_init(cls.__init__)

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


class UninitializedYamlObject:
    def __init__(self, cls, yaml_args):
        assert issubclass(cls, Serializable)
        self.cls = cls
        self.yaml_args = yaml_args
        self.obj = None

    def initialize(self, format_dict=None):
        """
        Initialize this object and all children.
        This object is the root of the reference tree
        :return: Serializable
        """
        # check args
        self._format_strings(format_dict or {})
        self._resolve_ref_default_args()
        self._create_referenced_default_args()
        self._share_init_params_top_down()
        self._init_components_bottom_up()
        return self.obj

    @property
    def init_arg_defaults(self):
        return inspect.signature(self.cls.__init__).parameters

    def _format_strings(self, format_dict):
        try:
            format_dict.update(_get_descendant(self, Path("exp_global.placeholders")))
        except PathError:
            pass
        if len(format_dict) == 0:
            return

        for path, node in _traverse_tree(self):
            if isinstance(node, str):
                try:
                    formatted = node.format(**format_dict)
                except (ValueError, KeyError, IndexError):
                    formatted = node
                if node != formatted:
                    _set_descendant(self, path, FormatString(formatted, node))
            elif isinstance(node, UninitializedYamlObject):
                init_args_defaults = node.init_arg_defaults
                for expected_arg in init_args_defaults:
                    if expected_arg not in [x[0] for x in _name_children(node)]:
                        arg_default = init_args_defaults[expected_arg].default
                        if isinstance(arg_default, str):
                            try:
                                formatted = arg_default.format(**format_dict)
                            except (ValueError, KeyError):  # will occur e.g. if a vocab entry contains a curly bracket
                                formatted = arg_default
                            if arg_default != formatted:
                                node.yaml_args[expected_arg] = FormatString(formatted, arg_default)

    def _resolve_ref_default_args(self):
        """
        If an argument is not given in the YAML file and is set to a Ref by default
        (e.g. dropout=Ref("exp_global.dropout")), copy this Ref into the implicit_args
        """
        for _, node in _traverse_tree(self):
            if isinstance(node, UninitializedYamlObject):
                init_args_defaults = node.init_arg_defaults
                for expected_arg in init_args_defaults:
                    if expected_arg not in node.yaml_args:
                        arg_default = init_args_defaults[expected_arg].default
                        if isinstance(arg_default, Ref):
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
                        raise ValueError(f"{shared_param_path} shared params {shared_param_set} contians Serializable"
                                         f"sub-object {child_of_shared_param} which is not permitted")
                if not isinstance(new_shared_val, Ref):
                    shared_val_choices.add(new_shared_val)
            if len(shared_val_choices) == 0:
                logger.warning(f"No param choices at for {shared_param_set}")
            elif len(shared_val_choices) > 1:
                logger.warning(f"inconsistent shared params for {shared_param_set}: "
                               f"{shared_val_choices}; Ignoring these shared parameters.")
            else:
                for shared_param_path in shared_param_set:
                    try:
                        if shared_param_path[-1] in \
                                _get_descendant(self, shared_param_path.parent()).init_arg_defaults:
                            _set_descendant(self, shared_param_path, list(shared_val_choices)[0])
                    except PathError:
                        # can happen when the shared path contained a reference,
                        # which we don't follow to avoid unwanted effects
                        pass

    def _init_components_bottom_up(self):
        cache = dict()
        for path, node in _traverse_tree_deep_once(self, self, _TraversalOrder.ROOT_LAST):
            if isinstance(node, Ref):
                cache_size = len(cache)
                initialized_component = self._resolve_ref(node, cache)
                if len(cache) == cache_size:
                    logger.debug(f"For {path}: reusing previously initialized {initialized_component}")
            elif isinstance(node, UninitializedYamlObject):
                for name, _ in _name_children(node):
                    initialized_child = cache[path.append(name)]
                    node.yaml_args[name] = initialized_child

                initialized_component = node._init_component()
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
            except PathError:
                print(resolved_path)
                if node.required:
                    # initialized_component = None
                    raise ReferenceError(str(resolved_path))
                else:
                    initialized_component = node.default
            cache[resolved_path] = initialized_component
            return initialized_component

    def _init_component(self):
        if self.obj is not None:
            return self.obj
        # check types
        assert not any(isinstance(param, UninitializedYamlObject) for param in self.yaml_args.values())
        obj = self.cls(**self.yaml_args)
        logger.debug(f"Initialized {self.cls.__name__}@{id(obj)}({self.yaml_args})"[:1000])
        self.obj = obj
        return obj


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

