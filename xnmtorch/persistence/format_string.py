from typing import Any

import yaml


class FormatString(str):

    __slots__ = ("unformatted",)

    def __new__(cls, value, unformatted) -> Any:
        obj = super().__new__(cls, value)
        obj.unformatted = unformatted
        return obj


def _represent_format_string(dumper: yaml.Dumper, obj: FormatString):
    dumper.represent_str(obj.unformatted)


yaml.add_representer(FormatString, _represent_format_string)
