use ggez::conf::{Backend, FullscreenType, NumSamples, WindowMode, WindowSetup};
use ggez::glam::Vec2;
use ggez::graphics::{Color, DrawParam, Image, InstanceArray, Rect, Transform};
use ggez::input::keyboard::{KeyCode, KeyInput};
use ggez::mint::{Point2, Vector2};
use ggez::*;
use kd_tree::KdTree;
use ndarray::Array;
use palette::rgb::Rgb;
use palette::{FromColor, Hsl, Hsv, LinSrgb, Mix};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use settings::*;

use clap::Parser;
use ndarray_ndimage::*;
use std::f32::consts::TAU;
use std::{env, path};

mod settings;
mod substrate;
mod useful;

mod family;

mod particle;
struct State {
    args: Args,
    timestep: f32,
    time: f32,
    overflow: f32,
    families: Vec<family::Family>,
    particles: Vec<particle::Particle>,
    rng: SmallRng,
    counts: Vec<usize>,
    blob_instance_array: InstanceArray,
    line_instance_array: InstanceArray,
    substrate: ndarray::Array2<f32>,
}

impl State {
    fn new(ctx: &mut Context, args: Args) -> GameResult<State> {
        let timestep = 1.0 / args.update_rate;
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
            substrate: Array::zeros((substrate::RESOLUTION, substrate::RESOLUTION)),
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
        let mut families: Vec<family::Family> = Vec::new();

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

        fn new_reaction(
            min_attact: f32,
            attract_mult: f32,
            force: f32,
            rng: &mut SmallRng,
        ) -> (f32, f32) {
            (
                min_attact + rng.gen::<f32>() * attract_mult,
                rng.gen::<f32>().abs().powf(0.3) * (-1.0 + rng.gen::<f32>() * 2.0).signum() * force,
            )
        }

        for i in 0..family_count {
            let color = Rgb::from_color(Hsv::<f32>::new(
                (i as f32 / family_count as f32) * 360.0 + h_offset,
                0.6,
                1.0,
            ))
            .into_components();

            families.push(family::Family {
                color: Color::from(color),
                id: i as usize,
                repel_range,
                repel_force,
                max_attraction,
                inflection: (
                    min_inflect + rng.gen::<f32>() * inflect_mult,
                    min_inflect + useful::smoothstep(rng.gen::<f32>()) * inflect_mult,
                ),
                inflection_curve: useful::smoothstep(rng.gen::<f32>()) * 10.0,
                reaction_freq: TAU * rng.gen::<f32>() * 0.02 + 0.01,
                reaction_phase: rng.gen::<f32>() * TAU,
                reactions_cold: (0..family_count)
                    .map(|_| {
                        (
                            new_reaction(min_attact, attract_mult, force, rng),
                            new_reaction(min_attact, attract_mult, force, rng),
                        )
                    })
                    .collect(),
                reactions_hot: (0..family_count)
                    .map(|_| {
                        (
                            new_reaction(min_attact, attract_mult, force, rng),
                            new_reaction(min_attact, attract_mult, force, rng),
                        )
                    })
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

        fn new_reaction(
            min_attact: f32,
            attract_mult: f32,
            force: f32,
            rng: &mut SmallRng,
        ) -> (f32, f32) {
            (
                min_attact + rng.gen::<f32>() * attract_mult,
                rng.gen::<f32>().abs().powf(0.3) * (-1.0 + rng.gen::<f32>() * 2.0).signum() * force,
            )
        }

        for i in 0..self.families.len() {
            let f = &self.families[i];
            self.families[i] = family::Family {
                color: f.color,
                id: f.id,
                repel_range: f.repel_range,
                repel_force: f.repel_force,
                max_attraction: f.max_attraction,
                inflection: (
                    min_inflect + rng.gen::<f32>() * inflect_mult,
                    min_inflect + useful::smoothstep(rng.gen::<f32>()) * inflect_mult,
                ),
                inflection_curve: useful::smoothstep(rng.gen::<f32>()) * 10.0,
                reaction_freq: TAU * rng.gen::<f32>() * 0.02 + 0.01,
                reaction_phase: rng.gen::<f32>() * TAU,
                reactions_cold: (0..family_count)
                    .map(|_| {
                        (
                            new_reaction(min_attact, attract_mult, force, rng),
                            new_reaction(min_attact, attract_mult, force, rng),
                        )
                    })
                    .collect(),
                reactions_hot: (0..family_count)
                    .map(|_| {
                        (
                            new_reaction(min_attact, attract_mult, force, rng),
                            new_reaction(min_attact, attract_mult, force, rng),
                        )
                    })
                    .collect(),
            };
        }
        // for i in 0..self.particles.len() {
        //     self.particles[i].family = self.rng.gen_range(0..self.families.len())
        // }
    }

    fn intialize_particles(&mut self) {
        let normal = Normal::new(768.0*0.5, 120.0).unwrap();
        let mut particles: Vec<particle::Particle> = vec![];
        let rng = &mut self.rng;
        let p_count = self.args.particle_count;

        println!("Families: {}", self.families.len());

        for _ in 0..p_count {
            let x = normal.sample(rng);
            let y = normal.sample(rng);

            particles.push(particle::Particle {
                position: Vec2 {
                    x,
                    y,
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
        self.substrate = Array::zeros((substrate::RESOLUTION, substrate::RESOLUTION));
    }
}

impl event::EventHandler<GameError> for State {
    fn key_down_event(&mut self, ctx: &mut Context, input: KeyInput, repeat: bool) -> GameResult {
        match input.keycode {
            Some(KeyCode::Escape) => ctx.request_quit(),
            Some(KeyCode::Space) => {
                if !repeat {
                    self.args.seed = None;
                    self.intialize_all()
                }
            }
            Some(KeyCode::R) => {
                if !repeat {
                    self.intialize_all();
                }
            }
            Some(KeyCode::F) => {
                if !repeat {
                    self.randomize_families();
                }
            }
            _ => (),
        }
        Ok(())
    }
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        let delta = ctx.time.delta().as_secs_f32() + self.overflow;

        let (actual_dt, step, step_count) = if delta > self.timestep {
            match self.args.catchup {
                SimulationCatchup::Slowdown => {
                    let actual_dt = self.timestep;
                    let step = self.timestep;
                    let step_count = (actual_dt / step) as usize;
                    (actual_dt, step, step_count)
                }
                SimulationCatchup::IgnoreUpdateRate => {
                    let actual_dt = delta;
                    let step = delta;
                    let step_count = 1;
                    (actual_dt, step, step_count)
                }
            }
        } else {
            let actual_dt = delta;
            let step = self.timestep;
            let step_count = (actual_dt / step) as usize;
            (actual_dt, step, step_count)
        };

        self.overflow = actual_dt - step * step_count as f32;

        for _ in 0..step_count {
            let dt = step;
            self.time += dt;
            let tree: KdTree<particle::Particle> =
                KdTree::par_build_by_ordered_float(self.particles.clone());
            (self.particles, self.counts) = self
                .particles
                .clone()
                .par_iter()
                .map(|p| p.update_velocity(&tree, self.time, dt, &self))
                .map(|p| (p.0.update_position(dt, &self), p.1))
                .collect();

            self.particles.iter().for_each(|particle| {
                let i = substrate::get_substrate_index(particle.position);
                self.substrate[i] += if particle.threat > 0.0 {
                    particle.threat * dt * 120.0
                } else {
                    particle.threat * dt
                }; // + particle.fear*dt;
            });

            let blurred = gaussian_filter(&self.substrate, dt * 15.0, 0, BorderMode::Reflect, 3);
            self.substrate = blurred;
            for i in 0..substrate::RESOLUTION {
                for j in 0..substrate::RESOLUTION {
                    let s = self.substrate[(i, j)];
                    self.substrate[(i, j)] =
                        (s - (s - 1.0).max(0.0) * 0.1 * dt * 0.0).max(0.0).min(4.0);
                }
            }
        }

        Ok(())
    }
    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = match self.args.transparent {
            ExplicitBoolean::Yes => {
                let mut canvas =
                    graphics::Canvas::from_frame(ctx, graphics::Color::from([0.0, 0.0, 0.0, 0.0]));
                canvas.set_blend_mode(graphics::BlendMode::ADD);
                canvas
            }
            ExplicitBoolean::No => {
                let mut canvas = graphics::Canvas::from_frame(
                    ctx,
                    graphics::Color::from([0.05, 0.025, 0.025, 1.0]),
                );
                canvas.set_blend_mode(graphics::BlendMode::ADD);
                canvas
            }
        };

        let size = ctx.gfx.size();
        let ratio = f32::min(size.0 / 768.0, size.1 / 768.0);
        let size = (size.0 / ratio, size.1 / ratio);
        let middle = (size.0 / 2.0, size.1 / 2.0);
        let scene = (768.0 / 2.0, 768.0 / 2.0);

        canvas.set_screen_coordinates(Rect {
            x: scene.0 - middle.0,
            y: scene.1 - middle.1,
            w: size.0,
            h: size.1,
        });

        self.blob_instance_array.clear();
        self.line_instance_array.clear();

        for particle in self.particles.iter() {
            let family = &self.families[particle.family];
            let vd = useful::safe_normalize(particle.velocity);
            let vm = particle.velocity.length();
            let phase_offset = useful::smoothstep(substrate::sample_substrate(
                particle.position,
                &self.substrate,
            ));
            let inflection_point =
                useful::lerp(family.inflection.0, family.inflection.1, phase_offset);
            let inflection_point = useful::lerp(inflection_point, 5.0, particle.fear);
            let inflection_curve = family.inflection_curve; // lerp(particle.family.inflection_curve, 3.0, self.fear);
            let vm = useful::inflect(vm / inflection_point, inflection_curve) * inflection_point;
            let velocity = vd * vm;
            let temp = substrate::sample_substrate(particle.position, &self.substrate);
            let threat = particle.threat * 0.5 + 0.5;
            let activity = threat.max(particle.fear);
            let life = ((velocity.length() - 1.0) * 0.1)
                .max(0.0)
                .powi(2)
                .clamp(0.0, 1.0);
            let flicker = ((ctx.time.time_since_start().as_secs_f32() * 5.0)
                + (particle.id as f32 % TAU))
                .sin();
            let sparkle = particle.fear * (1.0 - particle.threat).clamp(0.0, 1.0) * flicker;
            let sparkle = sparkle.clamp(0.0, 1.0).powf(2.0);

            let hot = LinSrgb::new(1.0, 0.5 * flicker, 0.0);
            let cold = LinSrgb::new(0.0, 0.5 * flicker, 1.0);

            let l = activity;

            let family_color = (family.color.r, family.color.g, family.color.b);
            let family_color = LinSrgb::new(family_color.0, family_color.1, family_color.2);
            let family_color_hsl = Hsl::from_color(family_color);
            let saturation = family_color_hsl.saturation;
            let lightness = family_color_hsl.lightness;
            let family_color = if temp > 0.5 {
                let interp = temp * 2.0 - 1.0;
                family_color.mix(hot, interp * 0.15)
            } else {
                let interp = 1.0 - (temp * 2.0);
                family_color.mix(cold, interp * 0.25)
            };
            let family_color = family_color.into_format();
            let mut family_color = Hsl::from_color(family_color);
            family_color.saturation = saturation;
            family_color.lightness = lightness;
            //let family_color = family_color.shift_hue(sub_val*180.0);
            //family_color.saturation = 1.0-sub_val;
            //let mut family_color = family_color; family_color.saturation = sub_val;
            let family_color = Rgb::from_color(family_color);
            let family_color =
                Color::from((family_color.red, family_color.green, family_color.blue));
            let color = (
                (particle.fear + sparkle).clamp(0.0, 1.0), //lerp(color.0 as f32 / 256.0, 1.0, threat),
                (threat + sparkle).clamp(0.0, 1.0), //lerp(color.1 as f32 / 256.0, 1.0, threat),
                (1.0 - activity + sparkle).clamp(0.0, 1.0), //lerp(color.2 as f32 / 256.0, 0.0, threat),
            );
            let color = Color::from((
                useful::lerp(family_color.r, color.0, l),
                useful::lerp(family_color.g, color.1, l),
                useful::lerp(family_color.b, color.2, l),
                useful::lerp(0.01, 1.0, life.max(l)),
                //lerp(0.0, 1.0, smoothstep(life.max(l).max(0.1))),
            ));
            //let scale = lerp(l.max(1.0-life), l.max(sparkle), sub_val);
            let scale = 1.0 - life;
            let scale = scale * scale;
            //let scale = smoothstep(scale);
            let scale = useful::lerp(0.03, 0.06, scale);
            self.blob_instance_array.push(
                DrawParam::new()
                    .offset(Vec2::new(256.0, 256.0))
                    .dest(Point2 {
                        x: particle.position.x + velocity.x * self.overflow.min(self.timestep),
                        y: particle.position.y + velocity.y * self.overflow.min(self.timestep),
                    })
                    .color(color)
                    //.rotation(gt*2.0)
                    //.scale(Vec2::new(0.03, 0.03)),
                    .scale(Vec2::new(scale, scale)),
                //.scale(Vec2::new(lerp(0.03,0.06,1.0-life), lerp(0.03,0.06,1.0-life))),
                //.scale(Vec2::new(lerp(0.03,0.06,sub_val), lerp(0.03,0.06,sub_val))),
            );
            for (line, _vel) in &particle.lines {
                //let scale = 0.1;
                let line = *line;
                // let line = Vec2 {
                //     x: line.x + vel.x*self.overflow.min(self.timestep),
                //     y: line.y + vel.y*self.overflow.min(self.timestep),
                // };
                let pos = Vec2 {
                    x: particle.position.x + velocity.x * self.overflow.min(self.timestep),
                    y: particle.position.y + velocity.y * self.overflow.min(self.timestep),
                };
                let diff = pos - line;
                let dist = (diff).length();
                if dist >= self.args.connecting_line_length {
                    continue;
                }
                let alpha = 1.0 - (dist / self.args.connecting_line_length);
                let alpha = alpha * alpha;
                let alpha = useful::smoothtable(alpha, 2.0, 2.0);
                let alpha = useful::smoothstep(alpha);
                //if alpha<0.2 || dist<0.1 { continue; }
                let mut color = color.clone();
                let alpha = alpha * color.a * 0.25;
                if alpha < 0.1 {
                    continue;
                }
                let angle = Vec2::angle_between(Vec2::new(0.0, 1.0), diff.normalize());
                color.a = alpha;
                //let pos = pos+(line*(1.0/3.0));
                self.line_instance_array.push(
                    DrawParam::new()
                        .offset(Vec2::new(67.5, 469.0))
                        .dest(pos)
                        .color(color)
                        .rotation(angle)
                        //.scale(Vec2::new(0.03, 0.03)),
                        .scale(Vec2::new(scale, (dist * 2.0) / 469.0)),
                    //.scale(Vec2::new(lerp(0.03,0.06,1.0-life), lerp(0.03,0.06,1.0-life))),
                    //.scale(Vec2::new(lerp(0.03,0.06,sub_val), lerp(0.03,0.06,sub_val))),
                );
            }
        }

        

        canvas.draw(&self.blob_instance_array, DrawParam::default());
        canvas.draw(&self.line_instance_array, DrawParam::default());

        let mut fps_text = format!("");
        let frame_time = 1.0 / ctx.time.fps() as f32;

        if ctx.keyboard.is_key_pressed(KeyCode::Tab) {
            if frame_time > self.timestep {
                fps_text.push_str(&format!(
                    "Simulation slowed to {}%\n",
                    ((self.timestep / frame_time) * 100.0) as usize
                ));
            }
            fps_text.push_str(&format!("Seed: {}\n", self.args.seed.unwrap_or(0)));
            fps_text.push_str(&format!("F/s: {}\n", ctx.time.fps() as u32));

            let counts: Vec<_> = self
                .counts
                //.par_iter()
                .iter()
                .map(|count| *count as f32)
                .collect();
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
            ctx.gfx
                .set_mode(WindowMode {
                    width: args.resolution.into(),
                    height: args.resolution.into(),
                    //maximized: true,
                    fullscreen_type,
                    borderless,
                    resizable: true,
                    transparent: args.transparent.into(),
                    //logical_size: Some(LogicalSize::new(768.0, 768.0)),
                    ..Default::default()
                })
                .unwrap();
        }
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
        }
        FullscreenType::Desktop => {
            match ctx
                .gfx
                .supported_resolutions()
                .max_by(|a, b| (a.width * a.height).cmp(&(b.width * b.height)))
            {
                Some(size) => {
                    ctx.gfx
                        .set_mode(WindowMode {
                            width: size.width as f32,
                            height: size.height as f32,
                            fullscreen_type: fullscreen_type,
                            transparent: args.transparent.into(),
                            ..Default::default()
                        })
                        .unwrap();
                }
                None => panic!("Could not set desktop fullscreen mode."),
            }
        }
    };

    let state = State::new(&mut ctx, args).unwrap();
    event::run(ctx, event_loop, state);
}
