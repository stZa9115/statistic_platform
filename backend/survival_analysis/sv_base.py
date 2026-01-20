from abc import ABC, abstractmethod

class svTest(ABC):
    name:str
    display_name:str
    result_prefix:str

    @abstractmethod
    def run(self,df):
        pass