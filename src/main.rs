use ggez::conf::{Backend, FullscreenType, NumSamples, WindowMode, WindowSetup};
use ggez::glam::Vec2;
use ggez::graphics::{Color, DrawParam, Image, InstanceArray, Rect, Transform};
use ggez::input::keyboard::{KeyCode, KeyInput};
use ggez::mint::{Point2, Vector2};
use ggez::*;
use kd_tree::{KdPoint, KdTree};
use ndarray::Array;
use palette::rgb::Rgb;
use palette::{FromColor, Hsl, Hsv, LinSrgb, Mix};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use std::f32::consts::TAU;
use std::hash::{Hash, Hasher};
use std::sync::atomic::AtomicUsize;
use std::{env, path};
use ndarray_ndimage::*;
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ExplicitBoolean {
    Yes,
    No,
}

impl From<ExplicitBoolean> for bool {
    fn from(value: ExplicitBoolean) -> Self {
        value==ExplicitBoolean::Yes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ConnectingLines {
    None,
    Family,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ScreenMode {
    Window,
    BorderlessWindow,
    Fullscreen,
    DesktopFullscreen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum SimulationCatchup {
    Slowdown,
    IgnoreUpdateRate,
}

/// Sparkle
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Seed to use
    #[arg(short, long)]
    seed: Option<u64>,

    /// Number of particles
    #[arg(short, long, default_value = "3000")]
    particle_count: u16,

    /// Number of families
    #[arg(short, long, default_value = "4")]
    family_count: u8,

    /// Simulation update rate
    #[arg(short, long, default_value = "30")]
    update_rate: f32,

    /// Limit rendering to display refresh rate
    #[arg(short, long, default_value = "yes")]
    vsync: ExplicitBoolean,

    /// Catchup strategy when rendering is slower than simulation
    #[arg(long, default_value = "slowdown")]
    catchup: SimulationCatchup,

    /// How to draw connecting lines
    #[arg(short, long, value_enum, default_value = "family")]
    connecting_lines: ConnectingLines,

    /// Length of connecting lines
    #[arg(short = 'l', long, default_value = "10")]
    connecting_line_length: f32,

    /// Display mode to use
    #[arg(short = 'm', long, value_enum, default_value = "desktop-fullscreen")]
    screen_mode: ScreenMode,

    /// In desktop and windowed modes, use a transparent background
    #[arg(short, long, default_value = "no")]
    transparent: ExplicitBoolean,
}

enum Seed {
    Entropy,
    U64(u64),
}

impl From<Option<u64>> for Seed {
    fn from(value: Option<u64>) -> Self {
        match value {
            Some(value) => Seed::U64(value),
            None => Seed::Entropy,
        }
    }
}

impl Seed {
    fn get_seed(&self) -> u64 {
        match self {
            Seed::Entropy => {
                let mut rng = SmallRng::from_entropy();
                rng.gen()
            },
            Seed::U64(u) => *u,
        }
    }
}

#[derive(Clone)]
struct Family {
    color: Color,
    id: usize,
    repel_range: f32,
    repel_force: f32,
    max_attraction: f32,
    reaction_freq: f32,
    reaction_phase: f32,
    reactions_cold: Vec<((f32, f32), (f32, f32))>,
    reactions_hot: Vec<((f32, f32), (f32, f32))>,
    inflection: (f32, f32),
    inflection_curve: f32,
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

#[derive(Clone)]
struct Particle {
    position: Vec2,
    velocity: Vec2,
    threat: f32,
    fear: f32,
    id: usize,
    family: usize,
    lines: Vec<Vec2>,
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

const SUBSTRATE_RESOLUTION: usize = 256;
struct State {
    args: Args,
    timestep: f32,
    time: f32,
    overflow: f32,
    families: Vec<Family>,
    particles: Vec<Particle>,
    rng: SmallRng,
    counts: Vec<usize>,
    blob_instance_array: InstanceArray,
    line_instance_array: InstanceArray,
    substrate: ndarray::Array2<f32>,
}

fn inflect(x: f32, curve: f32) -> f32 {
    let ease_out = |v: f32| {
        1.0 - 2.0f32.powf(-curve * v)
    };
    let a = ease_out(x);
    let b = x*(1.0-ease_out(1.0));
    a+b
}

fn get_downsample_position(position: Vec2, resolution: (usize, usize)) -> Vec2 {
    ((position.x/768.0)*resolution.0 as f32, (position.y/768.0)*resolution.1 as f32).into()
}

fn sample_substrate(position: Vec2, substrate: &ndarray::Array2<f32>) -> f32 {
    let pos = get_downsample_position(position, (SUBSTRATE_RESOLUTION, SUBSTRATE_RESOLUTION));
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
    bilerp(x1y1, x2y1, x1y2, x2y2, tx, ty)
}

fn get_substrate_index(position: Vec2) -> (usize, usize) {
    let index = (((position.x/768.0)*SUBSTRATE_RESOLUTION as f32) as usize, ((position.y/768.0)*SUBSTRATE_RESOLUTION as f32) as usize);
    (index.0.clamp(0, SUBSTRATE_RESOLUTION-1),index.1.clamp(0, SUBSTRATE_RESOLUTION-1))
}

fn bilerp(x1y1: f32, x2y1: f32, x1y2: f32, x2y2: f32, tx: f32, ty: f32) -> f32 {
    lerp(
        lerp(x1y1, x2y1, tx),
        lerp(x1y2, x2y2, tx),
        ty
    )
}

fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    //x*x*(3.0-2.0*x)
    x * x * x * (x * (6.0 * x - 15.0) + 10.0)
}

impl Particle {
    fn update_velocity(&self, tree: &KdTree<Particle>, gt: f32, dt: f32, state: &State) -> (Particle, usize) {
        let family = &state.families[self.family];
        let max_distance = family.repel_range + 2.0 * family.max_attraction;
        //let max_distance = self.family.max_attraction;
        let particles = tree.within_radius(
            &[self.position.x, self.position.y],
            max_distance,
        );

        // let push = tree.nearests(
        //     &[self.position.x, self.position.y],
        //     200,
        // );

        let count = AtomicUsize::new(0);
        let heat = smoothstep(sample_substrate(self.position, &state.substrate));
        let reaction_phase = smoothstep(f32::sin(family.reaction_freq*gt+family.reaction_phase)*0.5+0.5);
        let push = 
            particles
                .par_iter()
                // .filter_map(|item_and_distance| {
                //     if item_and_distance.squared_distance<max_distance {
                //         Some(item_and_distance.item)
                //     } else {
                //         None
                //     }
                // })
                .filter(|p| p.id != self.id)
                .map(|p| {
                    let p_family = &state.families[p.family];
                    count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let diff = p.position - self.position;
                    let dist = diff.length();
                    let reactions_hot = family.reactions_hot[p_family.id];
                    let reactions_cold = family.reactions_cold[p_family.id];
                    let reaction_hot = lerp_tuple(reactions_hot.0, reactions_hot.1, reaction_phase);
                    let reaction_cold = lerp_tuple(reactions_cold.0, reactions_cold.1, reaction_phase);
                    let reaction = lerp_tuple( reaction_cold,reaction_hot, heat);
                    let repel_range = family.repel_range;
                    let repel_force =
                        family.repel_force + 2.0 * self.fear * family.repel_force;
                    let reaction_range = reaction.0;
                    let reaction_force =
                        lerp(reaction.1, -1.0 * reaction.1.abs(), self.fear.powf(4.0));
                    let mult = if dist < family.repel_range {
                        (
                            (dist * repel_force) / repel_range - repel_force,
                            1.0 - (dist / repel_range),
                        )
                    } else if dist < repel_range + reaction_range {
                        (
                            ((reaction_force * dist - reaction_force * repel_range)
                                / reaction_range),
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
                    (mult.0 * safe_normalize(diff), mult.1)
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
                .map(|p| (p.position, &state.families[p.family]))
                .filter(|p| p.1.id == family.id)
                .filter_map(|p| if (p.0 - self.position).length_squared()<(connect_length*connect_length) {Some(p.0)} else {None})
                .collect(),
            ConnectingLines::All => particles
                .par_iter()
                .filter(|p| p.id != self.id)
                .filter(|p| p.id != self.id)
                .map(|p| (p.position, &state.families[p.family]))
                .filter_map(|p| if (p.0 - self.position).length_squared()<(connect_length*connect_length) {Some(p.0)} else {None})
                .collect(),
        };

        let offset = Vec2 {
            x: 768.0 / 2.0,
            y: 768.0 / 2.0,
        } - self.position;

        let gravity = offset;

        let grav_power = ((gravity.length()/150.0)-1.0).max(0.0);

        let gravity = safe_normalize(gravity) * grav_power.powi(2) * 60.0;

        let menace = if push.1 > 0.0 { push.1 } else { -1.0 };
        let push = push.0;

        let velocity = self.velocity
        + (self.velocity * -5.0
            + (safe_normalize(push) * push.length())//.min(150.0))
            + gravity
        )* dt;

        (Particle {
            position: self.position,
            velocity,
            threat: (self.threat + (menace * 0.0125 - 0.1) * dt).clamp(-1.0, 1.0),
            fear: (self.fear + self.threat * dt).clamp(0.0, 1.0),
            id: self.id,
            family: self.family.clone(),
            lines,
        },count.load(std::sync::atomic::Ordering::Relaxed))
    }
    fn update_position(self, dt: f32, state: &State) -> Particle {
        let family = &state.families[self.family];
        let vd = safe_normalize(self.velocity);
        let vm = self.velocity.length();
        let phase_offset = smoothstep(sample_substrate(self.position, &state.substrate));
        let inflection_point = lerp(family.inflection.0, family.inflection.1, phase_offset);
        let inflection_point = lerp(inflection_point, 5.0, self.fear);
        let inflection_curve = family.inflection_curve;// lerp(self.family.inflection_curve, 3.0, self.fear);
        let vm = inflect(vm/inflection_point, inflection_curve)*inflection_point;
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

fn safe_normalize(v: Vec2) -> Vec2 {
    if v.length_squared() < f32::EPSILON {
        v
    } else {
        v.normalize()
    }
}

fn lerp(a: f32, b: f32, n: f32) -> f32 {
    b * n + a * (1.0 - n)
}
fn lerp_tuple(a: (f32,f32), b: (f32,f32), n: f32) -> (f32,f32) {
    (lerp(a.0, b.0, n), lerp(a.1, b.1, n))
}

impl State {
    fn new(ctx: &mut Context, args: Args) -> GameResult<State> {
        let timestep = 1.0/args.update_rate;
        let mut state = State {
            args,
            timestep,
            time: 0.0,
            overflow: 0.0,
            families: vec![],
            particles: vec![],
            rng: SmallRng::from_entropy(),
            counts: vec![],
            blob_instance_array: InstanceArray::new(ctx, Image::from_path(ctx, "/blob.png")?),
            line_instance_array: InstanceArray::new(ctx, Image::from_path(ctx, "/line.png")?),
            substrate: Array::zeros((SUBSTRATE_RESOLUTION, SUBSTRATE_RESOLUTION)),
        };
        State::intialize_all(&mut state);

        Ok(state)
    }

    fn intialize_all(&mut self) {
        self.time = 0.0;
        self.intialize_seed();
        self.intialize_families();
        self.intialize_particles();
        self.intialize_substrate();
    }

    fn intialize_seed(&mut self) {
        let seed: Seed = self.args.seed.into();
        let seed = match seed {
            Seed::U64(u) => {
                let seed = u;
                println!("Using given seed: {seed}");
                seed
            }
            Seed::Entropy => {
                let seed = seed.get_seed();
                println!("Using random seed: {seed}");
                seed
            }
        };
        self.args.seed = Some(seed);
        self.rng = SmallRng::seed_from_u64(seed);
    }

    fn intialize_families(&mut self) {
        let rng = &mut self.rng;
        let mut families: Vec<Family> = Vec::new();

        let family_count = self.args.family_count;
        let repel_range = 10.0;
        let repel_force = 100.0;
        let min_attact = 2.0;
        let attract_mult = 10.0;
        let max_attraction = repel_range + 2.0 * (min_attact + attract_mult);
        let force = 50.0;
        let h_offset = rng.gen::<f32>() * 360.0;
        let min_inflect = 5.0;
        let inflect_mult = 10.0;

        fn new_reaction(min_attact: f32, attract_mult: f32, force: f32, rng: &mut SmallRng) -> (f32, f32) {
            (
                min_attact + rng.gen::<f32>() * attract_mult,
                rng.gen::<f32>().abs().powf(0.3) * (-1.0 + rng.gen::<f32>()*2.0).signum() * force,
            )
        }

        for i in 0..family_count {
            let color = Rgb::from_color(Hsv::<f32>::new(
                (i as f32 / family_count as f32) * 360.0 + h_offset,
                0.6,
                1.0,
            ))
            .into_components();

            families.push(Family {
                color: Color::from(color),
                id: i as usize,
                repel_range,
                repel_force,
                max_attraction,
                inflection: (min_inflect+rng.gen::<f32>()*inflect_mult, min_inflect+smoothstep(rng.gen::<f32>())*inflect_mult),
                inflection_curve: smoothstep(rng.gen::<f32>())*10.0,
                reaction_freq: TAU*rng.gen::<f32>()*0.02+0.01,
                reaction_phase: rng.gen::<f32>()*TAU,
                reactions_cold: (0..family_count)
                    .map(|_|
                        (
                            new_reaction(min_attact, attract_mult, force, rng), 
                            new_reaction(min_attact, attract_mult, force, rng)
                        )
                    )
                    .collect(),
                reactions_hot: (0..family_count)
                    .map(|_|
                        (
                            new_reaction(min_attact, attract_mult, force, rng), 
                            new_reaction(min_attact, attract_mult, force, rng)
                        )
                    )
                    .collect(),
            })
        }
        self.families = families;
        self.randomize_families();
    }

    fn randomize_families(&mut self) {
        let rng = &mut self.rng;

        let family_count = self.args.family_count;
        let min_inflect = 5.0;
        let inflect_mult = 10.0;
        let min_attact = 2.0;
        let attract_mult = 10.0;
        let force = 50.0;

        fn new_reaction(min_attact: f32, attract_mult: f32, force: f32, rng: &mut SmallRng) -> (f32, f32) {
            (
                min_attact + rng.gen::<f32>() * attract_mult,
                rng.gen::<f32>().abs().powf(0.3) * (-1.0 + rng.gen::<f32>()*2.0).signum() * force,
            )
        }

        for i in 0..self.families.len() {
            let f = &self.families[i];
            self.families[i] = Family {
                color: f.color,
                id: f.id,
                repel_range: f.repel_range,
                repel_force: f.repel_force,
                max_attraction: f.max_attraction,
                inflection: (min_inflect+rng.gen::<f32>()*inflect_mult, min_inflect+smoothstep(rng.gen::<f32>())*inflect_mult),
                inflection_curve: smoothstep(rng.gen::<f32>())*10.0,
                reaction_freq: TAU*rng.gen::<f32>()*0.02+0.01,
                reaction_phase: rng.gen::<f32>()*TAU,
                reactions_cold: (0..family_count)
                    .map(|_|
                        (
                            new_reaction(min_attact, attract_mult, force, rng), 
                            new_reaction(min_attact, attract_mult, force, rng)
                        )
                    )
                    .collect(),
                reactions_hot: (0..family_count)
                    .map(|_|
                        (
                            new_reaction(min_attact, attract_mult, force, rng), 
                            new_reaction(min_attact, attract_mult, force, rng)
                        )
                    )
                    .collect(),
            };
        }
        // for i in 0..self.particles.len() {
        //     self.particles[i].family = self.rng.gen_range(0..self.families.len())
        // }
    }

    fn intialize_particles(&mut self) {
        let mut particles: Vec<Particle> = vec![];
        let rng = &mut self.rng;
        let p_count = self.args.particle_count;

        println!("Families: {}", self.families.len());

        for _ in 0..p_count {
            let angle = TAU * rng.gen::<f32>();
            let dist = rng.gen::<f32>();
            let sdist = dist.sqrt();
            let sdist = sdist * 250.0;

            particles.push(Particle {
                position: Vec2 {
                    x: 768.0 * 0.5 + angle.sin() * sdist,
                    y: 768.0 * 0.5 + angle.cos() * sdist,
                },
                id: rng.gen(),
                velocity: Vec2 { x: 0.0, y: 0.0 },
                threat: -1.0,
                fear: 0.0,
                //family: Box::from(families[(dist * families.len() as f32) as usize].clone()),
                family: rng.gen_range(0..self.families.len()),
                lines: vec![],
            })
        }
        self.particles = particles;
    }

    fn intialize_substrate(&mut self) {
        self.substrate = Array::zeros((SUBSTRATE_RESOLUTION, SUBSTRATE_RESOLUTION));
    }

}

impl event::EventHandler<GameError> for State {
    fn key_down_event(&mut self, ctx: &mut Context, input: KeyInput, repeat: bool) -> GameResult {
        match input.keycode {
            Some(KeyCode::Escape) => ctx.request_quit(),
            Some(KeyCode::Space) => if !repeat {
                self.args.seed = None;
                self.intialize_all()
            },
            Some(KeyCode::R) => if !repeat {
                self.intialize_all();
            },
            Some(KeyCode::F) => if !repeat {
                self.randomize_families();
            },
            _ => (),
        }
        Ok(())
    }
    fn update(&mut self, ctx: &mut Context) -> GameResult {

        let delta = ctx.time.delta().as_secs_f32()+self.overflow;

        let (actual_dt, step, step_count) = if delta>self.timestep {
            match self.args.catchup {
                SimulationCatchup::Slowdown => {
                    let actual_dt = self.timestep;
                    let step = self.timestep;
                    let step_count = (actual_dt/step) as usize;
                    (actual_dt, step, step_count)
                },
                SimulationCatchup::IgnoreUpdateRate => {
                    let actual_dt = delta;
                    let step = delta;
                    let step_count = 1;
                    (actual_dt, step, step_count)
                },
            }
        } else {
            let actual_dt = delta;
            let step = self.timestep;
            let step_count = (actual_dt/step) as usize;
            (actual_dt, step, step_count)
        };

        self.overflow = actual_dt-step*step_count as f32;

        for _ in 0..step_count {
            let dt = step;
            self.time += dt;
            let tree: KdTree<Particle> = KdTree::par_build_by_ordered_float(self.particles.clone());
            (self.particles, self.counts) = self
                .particles
                .clone()
                .par_iter()
                .map(|p| p.update_velocity(&tree, self.time, dt, &self))
                .map(|p| (p.0.update_position(dt, &self), p.1))
                .collect();

            self.particles.iter().for_each(|particle|{
                let i = get_substrate_index(particle.position);
                self.substrate[i]+=if particle.threat>0.0 {particle.threat*dt*80.0} else {particle.threat*dt*5.0};// + particle.fear*dt;
            });

            let blurred = gaussian_filter(&self.substrate, dt*10.0, 0, BorderMode::Reflect, 3);
            self.substrate = blurred;
            for i in 0..SUBSTRATE_RESOLUTION {
                for j in 0..SUBSTRATE_RESOLUTION {
                    let s = self.substrate[(i,j)];
                    self.substrate[(i,j)] = (s-(s-1.0).max(0.0)*0.15*dt*0.0).max(0.0).min(3.0);
                }
            }
        }

        // let (fears, threats): (Vec<_>, Vec<_>) = self.particles
        //     .par_iter()
        //     .map(|particle| (particle.fear, particle.threat))
        //     .collect();
        
        // fn order(a: &&f32, b: &&f32) -> Ordering {
        //     if a>b {
        //         Ordering::Greater
        //     } else {
        //         Ordering::Less
        //     }
        // };
        // let fear = fears.par_iter().max_by(order).unwrap_or(&0.0);
        // let threat = threats.par_iter().max_by(order).unwrap_or(&0.0);

        // println!("{fear}\t{threat}");

        Ok(())
    }
    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let hot = LinSrgb::new(1.0, 0.5, 0.0);
        let cold = LinSrgb::new(0.0, 1.0, 0.5);
        let mut canvas = match self.args.transparent {
            ExplicitBoolean::Yes => {
                let mut canvas = graphics::Canvas::from_frame(ctx, graphics::Color::from([0.0, 0.0, 0.0, 0.0]));
                canvas.set_blend_mode(graphics::BlendMode::ADD);
                canvas
            },
            ExplicitBoolean::No => {
                let mut canvas = graphics::Canvas::from_frame(ctx, graphics::Color::from([0.05, 0.025, 0.025, 1.0]));
                canvas.set_blend_mode(graphics::BlendMode::ADD);
                canvas
            },
        };

        let size = ctx.gfx.size();
        let ratio = f32::min(size.0/768.0,size.1/768.0);
        let size = (size.0/ratio, size.1/ratio);
        let middle = (size.0/2.0, size.1/2.0);
        let scene = (768.0/2.0,768.0/2.0);

        canvas.set_screen_coordinates(Rect { x:scene.0-middle.0, y: scene.1-middle.1, w: size.0, h: size.1 });

        self.blob_instance_array.clear();
        self.line_instance_array.clear();

        for particle in self.particles.iter() {
            let family = &self.families[particle.family];
            let vd = safe_normalize(particle.velocity);
            let vm = particle.velocity.length();
            let phase_offset = smoothstep(sample_substrate(particle.position, &self.substrate));
            let inflection_point = lerp(family.inflection.0, family.inflection.1, phase_offset);
            let inflection_point = lerp(inflection_point, 5.0, particle.fear);
            let inflection_curve = family.inflection_curve;// lerp(particle.family.inflection_curve, 3.0, self.fear);
            let vm = inflect(vm/inflection_point, inflection_curve)*inflection_point;
            let velocity = vd * vm;
            let temp = sample_substrate(particle.position, &self.substrate);
            let threat = particle.threat * 0.5 + 0.5;
            let activity = threat.max(particle.fear);
            let life = ((velocity.length()-1.0)*0.1).max(0.0).powi(2).clamp(0.0, 1.0);
            let sparkle = ((ctx.time.time_since_start().as_secs_f32() * 5.0)
                + (particle.id as f32 % TAU))
                .sin();
            let sparkle = particle.fear * (1.0 - particle.threat).clamp(0.0, 1.0) * sparkle;
            let sparkle = sparkle.clamp(0.0, 1.0).powf(2.0);
            let l = activity;

            let family_color = (family.color.r, family.color.g, family.color.b);
            let family_color = LinSrgb::new(family_color.0, family_color.1, family_color.2);
            let family_color_hsl = Hsl::from_color(family_color);
            let saturation = family_color_hsl.saturation;
            let lightness = family_color_hsl.lightness;
            let family_color = if temp>0.5 {
                let interp = temp*2.0-1.0;
                family_color.mix(hot, interp*0.15)
            } else {
                let interp = 1.0-(temp*2.0);
                family_color.mix(cold, interp*0.25)
            };
            let family_color = family_color.into_format();
            let mut family_color = Hsl::from_color(family_color);
            family_color.saturation = saturation;
            family_color.lightness = lightness;
            //let family_color = family_color.shift_hue(sub_val*180.0);
            //family_color.saturation = 1.0-sub_val;
            //let mut family_color = family_color; family_color.saturation = sub_val;
            let family_color = Rgb::from_color(family_color);
            let family_color = Color::from((family_color.red, family_color.green, family_color.blue));
            let color = (
                (particle.fear + sparkle).clamp(0.0, 1.0), //lerp(color.0 as f32 / 256.0, 1.0, threat),
                (threat + sparkle).clamp(0.0, 1.0), //lerp(color.1 as f32 / 256.0, 1.0, threat),
                (1.0 - activity + sparkle).clamp(0.0, 1.0), //lerp(color.2 as f32 / 256.0, 0.0, threat),
            );
            let color = Color::from((
                lerp(family_color.r, color.0, l),
                lerp(family_color.g, color.1, l),
                lerp(family_color.b, color.2, l),
                //lerp(0.01, 1.0, life.max(l)),
                lerp(0.01, 1.0, life.max(l)),
            ));
            //let scale = lerp(l.max(1.0-life), l.max(sparkle), sub_val);
            let scale = 1.0-life;
            let scale = scale*scale;
            //let scale = smoothstep(scale);
            let scale = lerp(0.02,0.06,scale);
            self.blob_instance_array.push(
                DrawParam::new()
                    .offset(Vec2::new(256.0, 256.0))
                    .dest(Point2 {
                        x: particle.position.x + velocity.x*self.overflow.min(self.timestep),
                        y: particle.position.y + velocity.y*self.overflow.min(self.timestep),
                    })
                    .color(color)
                    //.rotation(gt*2.0)
                    //.scale(Vec2::new(0.03, 0.03)),
                    .scale(Vec2::new(scale, scale)),
                    //.scale(Vec2::new(lerp(0.03,0.06,1.0-life), lerp(0.03,0.06,1.0-life))),
                    //.scale(Vec2::new(lerp(0.03,0.06,sub_val), lerp(0.03,0.06,sub_val))),
            );
            for line in &particle.lines {
                //let scale = 0.1;
                let line = Vec2 { x: line.x, y: line.y };
                let pos = Vec2 {
                    x: particle.position.x + velocity.x*self.overflow.min(self.timestep),
                    y: particle.position.y + velocity.y*self.overflow.min(self.timestep),
                };
                let dist = ((pos-line).length())/self.args.connecting_line_length;
                let alpha = 1.0-dist;
                let alpha = smoothstep(alpha);
                //let alpha = alpha*alpha;
                //if alpha<0.2 || dist<0.1 { continue; }
                let mut color = color.clone();
                let alpha = alpha * color.a;
                if alpha<0.2 { continue; }
                let angle = Vec2::angle_between(Vec2::new(0.0, 1.0), pos-line);
                color.a = alpha * 0.5;
                //let pos = pos+(line*(1.0/3.0));
                self.line_instance_array.push(
                    DrawParam::new()
                        .offset(Vec2::new(67.5, 234.5*2.0))
                        .dest(pos)
                        .color(color)
                        .rotation(angle)
                        //.scale(Vec2::new(0.03, 0.03)),
                        .scale(Vec2::new( scale, dist/25.0)),
                        //.scale(Vec2::new(lerp(0.03,0.06,1.0-life), lerp(0.03,0.06,1.0-life))),
                        //.scale(Vec2::new(lerp(0.03,0.06,sub_val), lerp(0.03,0.06,sub_val))),
                );
            }
        }

        canvas.draw(&self.blob_instance_array, DrawParam::default());
        canvas.draw(&self.line_instance_array, DrawParam::default());

        let mut fps_text = format!("");
        let frame_time = 1.0 / ctx.time.fps() as f32;
        if frame_time>self.timestep {
            fps_text.push_str(&format!("Simulation slowed to {}%\n", ((self.timestep/frame_time)*100.0) as usize));    
        }

        if ctx.keyboard.is_key_pressed(KeyCode::Tab) {
            fps_text.push_str(&format!("Seed: {}\n", self.args.seed.unwrap_or(0)));
            fps_text.push_str(&format!("F/s: {}\n", ctx.time.fps() as u32));

            let counts: Vec<_> = self.counts.par_iter().map(|count| *count as f32).collect();
            let mean = statistical::mean(&counts);
            let median = statistical::median(&self.counts);
            let mode = statistical::mode(&self.counts).unwrap_or(0);
            let min = *self.counts.par_iter().min().unwrap_or(&0);
            let max = *self.counts.par_iter().max().unwrap_or(&0);

            fps_text.push_str(&format!("Min:\t{min}\n"));
            fps_text.push_str(&format!("Max:\t{max}\n"));
            fps_text.push_str(&format!("Mean:\t{}\n", mean as u32));
            fps_text.push_str(&format!("Median:\t{median}\n"));
            fps_text.push_str(&format!("Mode:\t{mode}\n"));
        }

        let fps = graphics::Text::new(fps_text);
            

        canvas.draw(
            &fps,
            DrawParam {
                src: Default::default(),
                color: Color::WHITE,
                transform: Transform::Values {
                    dest: Point2 { x: 20.0, y: 20.0 },
                    rotation: 0.0,
                    scale: Vector2 { x: 1.0, y: 1.0 },
                    offset: Point2 { x: 0.0, y: 0.0 },
                },
                z: 0,
            },
        );

        // let sub_divider = 768/SUBSTRATE_RESOLUTION;
        // let sub_scale = [sub_divider as f32, sub_divider as f32];
        // //let sub_scale = [2.0, 2.0];
        // for x in 0..SUBSTRATE_RESOLUTION {
        //     for y in 0..SUBSTRATE_RESOLUTION {
        //         if self.substrate[(x,y)]>0.8 {
        //             let xx = (x*sub_divider) as f32;
        //             let yy = (y*sub_divider) as f32;
        //             canvas.draw(&Quad, DrawParam::default().color(
        //                 Color::from((
        //                     1.0,
        //                     0.0,
        //                     1.0,
        //                     0.2,//self.substrate[(x,y)],
        //                 ))
        //             ).scale(sub_scale).dest([xx, yy]));
        //         }
        //     }
        // }

        canvas.finish(ctx)?;
        Ok(())
    }
}

pub fn main() {
    let args = Args::parse();
    let resource_dir = if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        let mut path = path::PathBuf::from(manifest_dir);
        path.push("resources");
        path
    } else {
        path::PathBuf::from("./resources")
    };

    let (fullscreen_type, borderless) = match args.screen_mode {
        ScreenMode::Window => (FullscreenType::Windowed, false),
        ScreenMode::BorderlessWindow => (FullscreenType::Windowed, true),
        ScreenMode::Fullscreen => (FullscreenType::True, false),
        ScreenMode::DesktopFullscreen => (FullscreenType::Desktop, false),
    };

    let (mut ctx, event_loop) = ContextBuilder::new("hello_ggez", "awesome_person")
        .window_mode(WindowMode {
            transparent: args.transparent.into(),
            ..Default::default()
        })
        .window_setup(WindowSetup {
            title: "Sparkle Particle Life Simulator".to_string(),
            samples: NumSamples::One,
            vsync: args.vsync.into(),
            icon: "".to_string(),
            srgb: false,
        })
        .add_resource_path(resource_dir)
        .backend(Backend::All)
        .build()
        .unwrap();

    match fullscreen_type {
        FullscreenType::Windowed => {
            ctx.gfx.set_mode(WindowMode {
                width: 768.0,
                height: 768.0,
                //maximized: true,
                fullscreen_type,
                borderless,
                resizable: true,
                transparent: args.transparent.into(),
                //logical_size: Some(LogicalSize::new(768.0, 768.0)),
                ..Default::default()
            }).unwrap();
        },
        FullscreenType::True => {
            match ctx.gfx.supported_resolutions().max_by(|a, b| {
                (a.width*a.height).cmp(&(b.width*b.height))
            }) {
                Some(size) => {
                    ctx.gfx.set_mode(WindowMode {
                        width: size.width as f32,
                        height: size.height as f32,
                        fullscreen_type: fullscreen_type,
                        transparent: args.transparent.into(),
                        ..Default::default()
                    }).unwrap();
                },
                None => panic!("Could not set true fullscreen mode. Try using \"-m desktop-fullscreen\" instead."),
            }
        },
        FullscreenType::Desktop => {
            match ctx.gfx.supported_resolutions().max_by(|a, b| {
                (a.width*a.height).cmp(&(b.width*b.height))
            }) {
                Some(size) => {
                    ctx.gfx.set_mode(WindowMode {
                        width: size.width as f32,
                        height: size.height as f32,
                        fullscreen_type: fullscreen_type,
                        transparent: args.transparent.into(),
                        ..Default::default()
                    }).unwrap();
                },
                None => panic!("Could not set desktop fullscreen mode."),
            }
        },
    };

    let state = State::new(&mut ctx, args).unwrap();
    event::run(ctx, event_loop, state);
}
