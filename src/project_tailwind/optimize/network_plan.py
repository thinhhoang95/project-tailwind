from typing import List, Union
from project_tailwind.optimize.regulation import Regulation


class NetworkPlan:
    """
    A NetworkPlan represents a collection of regulations that work together
    as a single network management strategy.
    """
    
    def __init__(self, regulations: Union[List[str], List[Regulation]] = None):
        """
        Initialize NetworkPlan with a list of regulations.
        
        Args:
            regulations: List of regulation strings or Regulation objects
        """
        self.regulations: List[Regulation] = []
        
        if regulations:
            for reg in regulations:
                if isinstance(reg, str):
                    self.regulations.append(Regulation(reg))
                elif isinstance(reg, Regulation):
                    self.regulations.append(reg)
                else:
                    raise TypeError("Regulations must be strings or Regulation objects")
    
    def add_regulation(self, regulation: Union[str, Regulation]):
        """Add a regulation to the network plan."""
        if isinstance(regulation, str):
            self.regulations.append(Regulation(regulation))
        elif isinstance(regulation, Regulation):
            self.regulations.append(regulation)
        else:
            raise TypeError("Regulation must be a string or Regulation object")
    
    def remove_regulation(self, index: int):
        """Remove a regulation by index."""
        if 0 <= index < len(self.regulations):
            self.regulations.pop(index)
        else:
            raise IndexError("Regulation index out of range")
    
    def clear(self):
        """Clear all regulations from the network plan."""
        self.regulations.clear()
    
    def __len__(self):
        """Return the number of regulations in the plan."""
        return len(self.regulations)
    
    def __iter__(self):
        """Allow iteration over regulations."""
        return iter(self.regulations)
    
    def __getitem__(self, index):
        """Allow indexing into regulations."""
        return self.regulations[index]
    
    def __str__(self):
        """String representation of the network plan."""
        if not self.regulations:
            return "NetworkPlan(empty)"
        
        reg_strs = [reg.raw_str for reg in self.regulations]
        return f"NetworkPlan({len(self.regulations)} regulations: {reg_strs})"
    
    def __repr__(self):
        return self.__str__()