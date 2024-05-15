use std::hash::Hasher;

use std::hash::Hash;

use ggez::graphics::Color;

#[derive(Clone)]
pub(crate) struct Family {
    pub(crate) color: Color,
    pub(crate) id: usize,
    pub(crate) repel_range: f32,
    pub(crate) repel_force: f32,
    pub(crate) max_attraction: f32,
    pub(crate) reaction_freq: f32,
    pub(crate) reaction_phase: f32,
    pub(crate) reactions_cold: Vec<((f32, f32), (f32, f32))>,
    pub(crate) reactions_hot: Vec<((f32, f32), (f32, f32))>,
    pub(crate) inflection: (f32, f32),
    pub(crate) inflection_curve: f32,
}

impl Hash for Family {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for Family {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Family {}
