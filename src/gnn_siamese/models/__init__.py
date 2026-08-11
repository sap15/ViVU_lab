"""Model components for the PKP2 siamese GNN project."""

from gnn_siamese.models.delta_block import (
    NodeDeltaBlock,
    NodeDeltaOutput,
    build_node_delta_features,
)
from gnn_siamese.models.encoder import EdgeAwareGraphEncoder, EncoderBranchOutput
from gnn_siamese.models.model import (
    ModelBContrastiveBaseline,
    ModelBContrastiveOutput,
    SharedSiameseEncoderModel,
    SharedSiameseEncoderOutput,
)
from gnn_siamese.models.model_a import (
    ModelANodalMultiscalePair,
    ModelANodalMultiscalePairOutput,
    ModelAOneView,
    ModelAOneViewOutput,
    ModelATwoView,
    ModelATwoViewOutput,
)
from gnn_siamese.models.multiscale_relational_a import (
    ModelAMultiscaleRelational,
    ModelAMultiscaleRelationalOutput,
    SCALE_ORDER_A,
    build_scale_relational,
)
from gnn_siamese.models.multiscale_pooling_a import (
    BranchMultiscalePooling,
    ModelAMultiscalePooling,
    ModelAMultiscalePoolingOutput,
    ScalePoolResult,
    aligned_selection_mask,
    indices_to_mask,
    segmented_pool,
    validate_delta_segmentation,
)
from gnn_siamese.models.projection import (
    InstanceProjectionHead,
    PairProjectionHead,
    ProjectionHeadConfig,
)
from gnn_siamese.models.projection_a import ModelAProjectionHead
from gnn_siamese.models.pair_fusion_a import MLPPairFusion
from gnn_siamese.models.relational import MLPDelta, RDelta, RelationalOutput, RelationalRepresentation

__all__ = [
    "EdgeAwareGraphEncoder",
    "EncoderBranchOutput",
    "InstanceProjectionHead",
    "MLPDelta",
    "MLPPairFusion",
    "ModelANodalMultiscalePair",
    "ModelANodalMultiscalePairOutput",
    "ModelAOneView",
    "ModelAOneViewOutput",
    "ModelATwoView",
    "ModelATwoViewOutput",
    "ModelAMultiscaleRelational",
    "ModelAMultiscaleRelationalOutput",
    "ModelBContrastiveBaseline",
    "ModelBContrastiveOutput",
    "ModelAMultiscalePooling",
    "ModelAMultiscalePoolingOutput",
    "ModelAProjectionHead",
    "BranchMultiscalePooling",
    "NodeDeltaBlock",
    "NodeDeltaOutput",
    "PairProjectionHead",
    "ProjectionHeadConfig",
    "RDelta",
    "RelationalOutput",
    "RelationalRepresentation",
    "SharedSiameseEncoderModel",
    "SharedSiameseEncoderOutput",
    "ScalePoolResult",
    "SCALE_ORDER_A",
    "aligned_selection_mask",
    "build_node_delta_features",
    "build_scale_relational",
    "indices_to_mask",
    "segmented_pool",
    "validate_delta_segmentation",
]
