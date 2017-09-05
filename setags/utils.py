from copy import deepcopy


class DictWrapper:
    @classmethod
    def from_dict(cls, source):
        ret = cls()
        ret.__dict__.update(source)
        return ret

    def as_dict(self) -> dict:
        return deepcopy(self.__dict__)
