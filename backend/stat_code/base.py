from abc import ABC, abstractmethod

class statTest(ABC):
    name:str
    display_name:str
    result_prefix:str

    @abstractmethod
    def run(self,df):
        pass