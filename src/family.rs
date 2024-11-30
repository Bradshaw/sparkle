use std::hash::Hasher;

use std::hash::Hash;

use ggez::graphics::Color;
use rand::rngs::SmallRng;
use rand::Rng;

use crate::useful;

#[derive(Clone)]
pub(crate) struct Family {
    pub(crate) color: Color,
    pub(crate) id: usize,
    pub(crate) repel_range: f32,
    pub(crate) repel_force: f32,
    pub(crate) max_attraction: f32,
    pub(crate) reaction_freq: f32,
    pub(crate) reaction_phase: f32,
    pub(crate) reactions_cold: Vec<PhasedReaction>,
    pub(crate) reactions_hot: Vec<PhasedReaction>,
    pub(crate) inflection: (f32, f32),
    pub(crate) inflection_curve: f32,
}

#[derive(Clone)]
pub(crate) struct Reactions {
    freq: f32,
    phase: f32,
    reactions: Vec<((Reaction, Reaction), (Reaction, Reaction))>,
}

impl Reactions {
    pub(crate) fn new(freq: f32, phase: f32) -> Self {
        todo!()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Reaction {
    pub(crate) range: f32,
    pub(crate) force: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct PhasedReaction (pub(crate) Reaction, pub(crate) Reaction);

impl Reaction {
    pub(crate) fn new(range: f32, force: f32) -> Self {
        Self {
            range,
            force
        }
    }

    pub(crate) fn random_reaction(
        min_attact: f32,
        attract_mult: f32,
        force: f32,
        rng: &mut SmallRng,
    ) -> Reaction {
        Reaction::new(min_attact + rng.gen::<f32>() * attract_mult,rng.gen::<f32>().abs().powf(0.3) * (-1.0 + rng.gen::<f32>() * 2.0).signum() * force)
    }

    pub(crate) fn lerp(&self, other: Self, n: f32) -> Self {
        Self {
            range: useful::lerp(self.range, other.range, n),
            force: useful::lerp(self.force, other.force, n),
        }
    }
}

impl PhasedReaction {
    pub(crate) fn sample(&self, n: f32) -> Reaction {
        self.0.lerp(self.1, n)
    }
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
