use rayon::prelude::*;
use std::sync::atomic::AtomicUsize;

use crate::{substrate, useful, ConnectingLines};

use super::State;

use kd_tree::{KdPoint, KdTree};

use ggez::glam::Vec2;

#[derive(Clone)]
pub(crate) struct Particle {
    pub(crate) position: Vec2,
    pub(crate) velocity: Vec2,
    pub(crate) threat: f32,
    pub(crate) fear: f32,
    pub(crate) id: usize,
    pub(crate) family: usize,
    pub(crate) lines: Vec<(Vec2, Vec2)>,
}

impl PartialEq for Particle {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Particle {}

impl KdPoint for Particle {
    type Scalar = f32;
    type Dim = typenum::U2;

    fn at(&self, i: usize) -> Self::Scalar {
        if i == 0 {
            self.position.x
        } else {
            self.position.y
        }
    }
}

impl Particle {
    pub(crate) fn update_velocity(
        &self,
        tree: &KdTree<Particle>,
        gt: f32,
        dt: f32,
        state: &State,
    ) -> (Particle, usize) {
        let family = &state.families[self.family];
        let max_distance = family.repel_range + 2.0 * family.max_attraction;
        //let max_distance = self.family.max_attraction;
        let particles = tree.within_radius(&[self.position.x, self.position.y], max_distance);

        // let push = tree.nearests(
        //     &[self.position.x, self.position.y],
        //     200,
        // );

        let count = AtomicUsize::new(0);
        let heat = useful::smoothstep(substrate::sample_substrate(self.position, &state.substrate));
        let reaction_phase = useful::smoothstep(
            f32::sin(family.reaction_freq * gt + family.reaction_phase) * 0.5 + 0.5,
        );
        let push = particles
            .par_iter()
            .filter(|p| p.id != self.id)
            .map(|p| {
                let p_family = &state.families[p.family];
                count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let diff = p.position - self.position;
                let dist = diff.length();
                let reactions_hot = family.reactions_hot[p_family.id];
                let reactions_cold = family.reactions_cold[p_family.id];
                let reaction_hot =
                    useful::lerp_tuple(reactions_hot.0, reactions_hot.1, reaction_phase);
                let reaction_cold =
                    useful::lerp_tuple(reactions_cold.0, reactions_cold.1, reaction_phase);
                let reaction = useful::lerp_tuple(reaction_cold, reaction_hot, heat);
                let repel_range = family.repel_range;
                let repel_force = family.repel_force + 2.0 * self.fear * family.repel_force;
                let reaction_range = reaction.0;
                let reaction_force =
                    useful::lerp(reaction.1, -1.0 * reaction.1.abs(), self.fear.powf(4.0));
                let mult = if dist < family.repel_range {
                    (
                        (dist * repel_force) / repel_range - repel_force,
                        1.0 - (dist / repel_range),
                    )
                } else if dist < repel_range + reaction_range {
                    (
                        ((reaction_force * dist - reaction_force * repel_range) / reaction_range),
                        0.0,
                    )
                } else if dist < repel_range + 2.0 * reaction_range {
                    (
                        (2.0 * reaction_force
                            + (reaction_force * repel_range - reaction_force * dist)
                                / reaction_range),
                        0.0,
                    )
                } else {
                    (0.0, 0.0)
                };
                (mult.0 * useful::safe_normalize(diff), mult.1)
            })
            .reduce(
                || (Vec2 { x: 0.0, y: 0.0 }, 0.0),
                |a, b| (a.0 + b.0, a.1 + b.1),
            );

        let connect_length = state.args.connecting_line_length;
        let lines: Vec<_> = match state.args.connecting_lines {
            ConnectingLines::None => vec![],
            ConnectingLines::Family => particles
                .par_iter()
                .filter(|p| p.id != self.id)
                .map(|p| (p.position, p.velocity, &state.families[p.family]))
                .filter(|p| p.2.id == family.id)
                .filter_map(|p| {
                    if (p.0 - self.position).length_squared() < (connect_length * connect_length) {
                        Some((p.0, p.1))
                    } else {
                        None
                    }
                })
                .collect(),
            ConnectingLines::All => particles
                .par_iter()
                .filter(|p| p.id != self.id)
                .map(|p| (p.position, p.velocity, &state.families[p.family]))
                .filter_map(|p| {
                    if (p.0 - self.position).length_squared() < (connect_length * connect_length) {
                        Some((p.0, p.1))
                    } else {
                        None
                    }
                })
                .collect(),
        };

        let offset = Vec2 {
            x: 768.0 / 2.0,
            y: 768.0 / 2.0,
        } - self.position;

        let gravity = offset;

        let grav_power = ((gravity.length() / 150.0) - 1.0).max(0.0);

        let gravity = useful::safe_normalize(gravity) * grav_power.powi(2) * 60.0;

        let menace = if push.1 > 0.0 { push.1 } else { -1.0 };
        let push = push.0;

        let velocity = self.velocity
            + (self.velocity * -5.0
            + (useful::safe_normalize(push) * push.length())//.min(150.0))
            + gravity)
                * dt;

        (
            Particle {
                position: self.position,
                velocity,
                threat: (self.threat + (menace * 0.0125 - 0.1) * dt).clamp(-1.0, 1.0),
                fear: (self.fear + self.threat * dt).clamp(0.0, 1.0),
                id: self.id,
                family: self.family.clone(),
                lines,
            },
            count.load(std::sync::atomic::Ordering::Relaxed),
        )
    }
    pub(crate) fn update_position(self, dt: f32, state: &State) -> Particle {
        let family = &state.families[self.family];
        let vd = useful::safe_normalize(self.velocity);
        let vm = self.velocity.length();
        let phase_offset =
            useful::smoothstep(substrate::sample_substrate(self.position, &state.substrate));
        let inflection_point = useful::lerp(family.inflection.0, family.inflection.1, phase_offset);
        let inflection_point = useful::lerp(inflection_point, 5.0, self.fear);
        let inflection_curve = family.inflection_curve; // lerp(self.family.inflection_curve, 3.0, self.fear);
        let vm = useful::inflect(vm / inflection_point, inflection_curve) * inflection_point;
        let velocity = vd * vm;

        let position = self.position + velocity * dt;
        Particle {
            position,
            velocity: self.velocity,
            threat: self.threat,
            fear: self.fear,
            id: self.id,
            family: self.family,
            lines: self.lines,
        }
    }
}
