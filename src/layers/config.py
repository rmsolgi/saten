from typing import Optional, Union, List, Callable, Any
from pydantic import BaseModel, ValidationError, field_validator, model_validator
from src.layers import utils



class LayerConfig(BaseModel):
    layer_class: Optional[Callable]=None
    mode: Optional[Union[str,None]]
    layer_name: Optional[str] = None
    info_path: Optional[str] = None
    rank: Optional[int] = None
    weight_transpose: Optional[bool] = None
    covariance_matrix: Optional[Any] = None

    in_order: Optional[int] = None
    out_order: Optional[int] = None
    decomp_error: Optional[float] = None
    sparsity: Optional[str] = None
    sparsity_ratio: Optional[list[float]] = None


    threshold: Optional[float] = None
    sparsity: Optional[str] = None
    decomp_format: Optional[str] = None
    m_n_ratio: Optional[list[int]] = None 

    freez: Optional[bool] = None

