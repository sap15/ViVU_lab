"""Model components for the PKP2 siamese GNN project."""

from gnn_siamese.models.encoder import EdgeAwareGraphEncoder, EncoderBranchOutput
from gnn_siamese.models.model import (
    ModelBContrastiveBaseline,
    ModelBContrastiveOutput,
    SharedSiameseEncoderModel,
    SharedSiameseEncoderOutput,
)
from gnn_siamese.models.projection import (
    InstanceProjectionHead,
    PairProjectionHead,
    ProjectionHeadConfig,
)
from gnn_siamese.models.relational import MLPDelta, RDelta, RelationalOutput, RelationalRepresentation

__all__ = [
    "EdgeAwareGraphEncoder",
    "EncoderBranchOutput",
    "InstanceProjectionHead",
    "MLPDelta",
    "ModelBContrastiveBaseline",
    "ModelBContrastiveOutput",
    "PairProjectionHead",
    "ProjectionHeadConfig",
    "RDelta",
    "RelationalOutput",
    "RelationalRepresentation",
    "SharedSiameseEncoderModel",
    "SharedSiameseEncoderOutput",
]
