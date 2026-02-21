# SCZ Xenium analysis modules
"""
Core analysis modules for the SCZ Xenium spatial transcriptomics pipeline.

Modules:
    loading         - Xenium data I/O (h5, boundaries)
    label_transfer  - Correlation-based label transfer from SEA-AD reference
    depth_model     - Cortical depth prediction + OOD scoring
    spatial_domains - Pia/meninges/vascular domain classification
    cell_qc         - Cell quality control filtering
    metadata        - Sample metadata and diagnosis mapping
    plotting        - Visualization utilities
    analysis        - Statistical testing
    layers          - Layer assignment utilities
    bundle_viewer   - Standalone HTML viewer bundler
"""
