from abc import ABC, abstractmethod
from typing import Any, List, Union
import pandas as pd

# Używamy Union[pd.DataFrame, Any] aby zachować elastyczność, 
# ale z uwagi na dane tabelaryczne, DF jest preferowane.

class BasePreProcessor(ABC):
    """
    Ogólny Interfejs Bazowy dla wszystkich kroków preprocessingu. 
    Wymusza standardowe metody fit/transform.
    """
    
    def __init__(self):
        # Atrybut, w którym będą przechowywane nazwy kolumn po transformacji.
        self._column_names: List[str] = []

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, Any] = None) -> 'BasePreProcessor':
        """Uczy parametry transformacji na danych X."""
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Stosuje nauczoną transformację na danych X."""
        pass
    
    # Metoda z Scikit-learn dla wygody
    def fit_transform(self, X: pd.DataFrame, y: Union[pd.Series, Any] = None) -> pd.DataFrame:
        """Uczy i stosuje transformację w jednym kroku."""
        return self.fit(X, y).transform(X)

    @abstractmethod
    def get_params(self) -> dict:
        """Zwraca parametry (hiperparametry) zaimplementowanego kroku."""
        pass
    
    def get_feature_names(self) -> List[str]:
        """Zwraca nazwy kolumn po transformacji przez ten krok."""
        return self._column_names

    # Dwie dodatkowe metody dla Auto ML i inspekcji
    @abstractmethod
    def get_description(self) -> str:
        """Zwraca krótki opis transformacji."""
        pass

class ValueModifier(BasePreProcessor):
    """
    Interfejs dla transformacji modyfikujących wartości w kolumnach 
    (np. skalowanie, imputacja, logarytmowanie).
    Nie zmienia liczby ani nazw kolumn.
    """
    # Wymaga się dziedziczenia i implementacji metod z BasePreProcessor
    # nie trzeba ich tu powtarzać, ale można dodać konkretne dla modyfikacji
    pass

class ShapeChanger(BasePreProcessor):
    """
    Interfejs dla transformacji zmieniających kształt (liczbę lub nazwy kolumn).
    Wymaga implementacji zarządzania kolumnami wejściowymi i wyjściowymi.
    """

    def __init__(self):
        super().__init__()
        # Inne atrybuty do śledzenia zmian
        self._input_column_names: List[str] = []
        self._new_column_names: List[str] = []
        self._removed_column_names: List[str] = []

    @abstractmethod
    def get_original_columns(self) -> List[str]:
        """Zwraca listę kolumn, które weszły do transformacji."""
        return self._input_column_names

    @abstractmethod
    def get_newly_created_columns(self) -> List[str]:
        """Zwraca listę kolumn, które zostały stworzone przez transformację."""
        return self._new_column_names

    @abstractmethod
    def get_removed_columns(self) -> List[str]:
        """Zwraca listę kolumn, które zostały usunięte przez transformację (jeśli dotyczy)."""
        return self._removed_column_names
        
    # Wymuszenie, aby klasa implementująca pamiętała o aktualizacji nazw w metodzie fit i transform
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Stosuje transformację. Musi ustawić self._column_names."""
        pass