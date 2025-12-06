from typing import runtime_checkable, Protocol, Iterator, TypeVar
from torch import Tensor

@runtime_checkable
class DataGeneratorProtocol(Protocol):
    """
    Common protocol for data generators used in Daisy.
    """
    def reset(self) -> None: ...
    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]: ...
    def __next__(self) -> tuple[Tensor, Tensor]: ...