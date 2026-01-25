from typing import Dict, Iterable, List, Optional, Union


class SimpleVocab:
    """
    轻量级替代 torchtext.vocab.Vocab，满足本项目用到的最小接口：
    - `__contains__`, `__getitem__`, `__len__`
    - `__call__` 映射一组 token -> indices
    - `append_token`, `insert_token`
    - `set_default_index`, `get_stoi`
    """

    def __init__(
        self,
        tokens: Optional[List[str]] = None,
        specials: Optional[List[str]] = None,
        special_first: bool = True,
    ) -> None:
        self._itos: List[str] = []
        self._stoi: Dict[str, int] = {}
        self._default_index: Optional[int] = None

        tokens = tokens or []
        specials = specials or []

        ordered: List[str] = []
        if specials:
            ordered.extend(specials if special_first else [])
        ordered.extend(tokens)
        if specials and not special_first:
            ordered.extend(specials)

        for tok in ordered:
            if tok not in self._stoi:
                self._stoi[tok] = len(self._itos)
                self._itos.append(tok)

    def __contains__(self, token: str) -> bool:
        return token in self._stoi

    def __len__(self) -> int:
        return len(self._itos)

    def __getitem__(self, token: Union[str, int]) -> Union[int, str]:
        if isinstance(token, int):
            return self._itos[token]
        # string -> index
        if token in self._stoi:
            return self._stoi[token]
        if self._default_index is not None:
            return self._default_index
        raise KeyError(f"Token '{token}' not in vocabulary and no default index set.")

    def __call__(self, tokens: Iterable[str]) -> List[int]:
        return [self.__getitem__(t) for t in tokens]

    def append_token(self, token: str) -> None:
        if token not in self._stoi:
            self._stoi[token] = len(self._itos)
            self._itos.append(token)

    def insert_token(self, token: str, index: int) -> None:
        # 仅用于从 dict 恢复时，确保是连续索引
        if token in self._stoi:
            return
        # 扩容到 index
        while len(self._itos) < index:
            placeholder = f"<unk_{len(self._itos)}>"
            self._itos.append(placeholder)
            self._stoi[placeholder] = len(self._itos) - 1
        if len(self._itos) == index:
            self._itos.append(token)
        else:
            self._itos[index] = token
        self._stoi[token] = index

    def set_default_index(self, index: int) -> None:
        self._default_index = index

    def get_stoi(self) -> Dict[str, int]:
        return dict(self._stoi)