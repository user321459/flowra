"""I/O Utilities"""
import torch
from typing import Dict, Any

def save_adapter(adapters: Dict[str, Any], path: str, metadata: Dict = None):
    """Save adapters to file."""
    state = {'adapters': {k: v.state_dict() for k, v in adapters.items()}}
    if metadata:
        state['metadata'] = metadata
    torch.save(state, path)

def load_adapter(path: str, adapters: Dict[str, Any] = None) -> Dict:
    """Load adapters from file."""
    state = torch.load(path, map_location='cpu')
    if adapters:
        for name, adapter_state in state.get('adapters', {}).items():
            if name in adapters:
                adapters[name].load_state_dict(adapter_state)
    return state
