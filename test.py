import inspect

import yaml
from torch import nn

from xnmtorch.persistence.serializable import Serializable, Ref


class Repr:
    def __str__(self):
        members = inspect.getmembers(self, lambda x: not inspect.ismethod(x) and not inspect.isfunction(x))
        cls_name = self.__class__.__name__
        arg_string = ", ".join(f"{member[0]}={member[1]}" for member in members
                               if not member[0].startswith("_"))
        return f"{cls_name}({arg_string})"


class Args(Serializable, Repr):
    def __init__(self, some, argument, test, third=0.5):
        self.some = some
        self.argument = argument
        self.test = test
        self.third = third


class SomeObject(Serializable, Repr):
    pass


class Project(Serializable, Repr):
    def __init__(self, args, some=Ref("args.some", 0.5), other="arg"):
        self.args = args
        self.some = some
        self.other = other


class SomeModule(Serializable, nn.Module):
    def __init__(self, layers, dropout=Ref("args.third")):
        super().__init__()
        self.layers = nn.Sequential(*[nn.Linear(10, 10) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        return self.dropout(self.layers(inputs))


text = """
!Project
    args: !Args
        some: 5
        argument: !SomeObject {}
        test: 2
    other: !Ref
        path: args.argument
"""

a_obj = Serializable.load(text)

print(a_obj)
print(a_obj.dump())
print(Serializable.load(a_obj.dump()))

args = SomeObject()
a = Project(Args(some=5, argument=args, test=2), other=args)
print(a)

text2 = """
!Project
    args: !Args
        some: 0.2
        argument: 0.7
        test: 3
    other:
        !SomeModule
            layers: 3
"""

module = Serializable.load(text2)
print(module)
print(module.dump())
print(module.clone())
