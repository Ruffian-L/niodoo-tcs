#![no_main]
use libfuzzer_sys::fuzz_target;
use nalgebra::DMatrix;
use tcs_ml::EquivariantLayer;

fuzz_target!(|data: (usize, usize, Vec<f32>)| {
    let (nrows, ncols, mut buf) = data;
    let nrows = (nrows % 8).max(1);
    let ncols = (ncols % 8).max(1);
    let need = nrows * ncols;
    if buf.len() < need { buf.resize(need, 0.0); }
    let positions = DMatrix::<f32>::from_row_slice(nrows, ncols, &buf[..need]);
    let features = DMatrix::<f32>::identity(nrows, nrows.min(ncols));

    let layer = EquivariantLayer::new(features.ncols(), features.ncols().max(1));
    let _ = layer.forward(&positions, &features);
});

