use ggez::glam::Vec2;
use ndarray::Array;

use crate::useful;

pub(crate) const RESOLUTION: usize = 256;

fn get_downsample_position(position: Vec2, resolution: (usize, usize)) -> Vec2 {
    (
        (position.x / 768.0) * resolution.0 as f32,
        (position.y / 768.0) * resolution.1 as f32,
    )
        .into()
}

pub(crate) fn sample_substrate(position: Vec2, substrate: &ndarray::Array2<f32>) -> f32 {
    let pos = get_downsample_position(position, (RESOLUTION, RESOLUTION));
    let xmin = pos.x.floor() as usize;
    let xmax = pos.x.ceil() as usize;
    let ymin = pos.y.floor() as usize;
    let ymax = pos.y.ceil() as usize;
    let x1y1 = *substrate.get((xmin, ymin)).unwrap_or(&0.0);
    let x2y1 = *substrate.get((xmax, ymin)).unwrap_or(&0.0);
    let x1y2 = *substrate.get((xmin, ymax)).unwrap_or(&0.0);
    let x2y2 = *substrate.get((xmax, ymax)).unwrap_or(&0.0);
    let tx = pos.x.fract();
    let ty = pos.y.fract();
    useful::bilerp(x1y1, x2y1, x1y2, x2y2, tx, ty)
}

pub(crate) fn get_substrate_index(position: Vec2) -> (usize, usize) {
    let index = (
        ((position.x / 768.0) * RESOLUTION as f32) as usize,
        ((position.y / 768.0) * RESOLUTION as f32) as usize,
    );
    (
        index.0.clamp(0, RESOLUTION - 1),
        index.1.clamp(0, RESOLUTION - 1),
    )
}

pub(crate) struct Substrate (ndarray::Array2<f32>);

impl Substrate {
    pub(crate) fn new() -> Self {
        Self(Array::zeros((RESOLUTION, RESOLUTION)))
    }
    pub(crate) fn sample(&self, at: Vec2) -> f32 {
        let pos = get_downsample_position(at, (RESOLUTION, RESOLUTION));
        let xmin = pos.x.floor() as usize;
        let xmax = pos.x.ceil() as usize;
        let ymin = pos.y.floor() as usize;
        let ymax = pos.y.ceil() as usize;
        let x1y1 = *self.0.get((xmin, ymin)).unwrap_or(&0.0);
        let x2y1 = *self.0.get((xmax, ymin)).unwrap_or(&0.0);
        let x1y2 = *self.0.get((xmin, ymax)).unwrap_or(&0.0);
        let x2y2 = *self.0.get((xmax, ymax)).unwrap_or(&0.0);
        let tx = pos.x.fract();
        let ty = pos.y.fract();
        useful::bilerp(x1y1, x2y1, x1y2, x2y2, tx, ty)
    }
}