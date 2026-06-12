//! Inspection, parameter, FLOPs, diagram, and shape-checking utilities.

mod diagram;
mod flops;
mod inspection;
mod params;
mod shape;

pub use diagram::{build_architecture_diagram, build_layer_table};
pub use flops::{
    estimate_detailed_flops, estimate_flops, flops_summary_text, format_flops,
    DetailedFlopsEstimate, FlopsEstimate,
};
pub use inspection::{architecture_text, inspection_markdown, LayerReport, ModelInspection};
pub use params::{
    format_parameter_count, parameter_summary_text, ParameterBreakdown, ParameterReport,
};
pub use shape::{verify_shapes, ShapeVerificationReport};
