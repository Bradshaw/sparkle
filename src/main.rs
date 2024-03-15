use ggez::conf::{Backend, FullscreenType, NumSamples, WindowMode, WindowSetup};
use ggez::glam::Vec2;
use ggez::graphics::{Color, DrawParam, Image, InstanceArray, Quad, Rect, Transform};
use ggez::mint::{Point2, Vector2};
use ggez::winit::dpi::LogicalSize;
use ggez::*;
use kd_tree::{KdPoint, KdTree};
use ndarray::{Array, Array2, AsArray, FixedInitializer};
use palette::rgb::Rgb;
use palette::{FromColor, Hsv};
use rand::Rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::cmp::Ordering;
use std::f32::consts::{PI, TAU};
use std::hash::{Hash, Hasher};
use std::sync::atomic::AtomicUsize;
use std::{env, path};
use ndarray_ndimage::*;

#[derive(Clone)]
struct Family {
    color: Color,
    id: usize,
    repel_range: f32,
    repel_force: f32,
    max_attraction: f32,
    reaction_freq: f32,
    reaction_phase: f32,
    reactions_a: Vec<(f32, f32)>,
    reactions_b: Vec<(f32, f32)>,
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
    family: Box<Family>,
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

const SUBSTRATE_RESOLUTION: usize = 64;
struct State {
    particles: Vec<Particle>,
    counts: Vec<usize>,
    instance_array: InstanceArray,
    substrate: ndarray::Array2<f32>,
}

fn get_substrate_index(position: Vec2) -> (usize, usize) {
    (((position.x/768.0)*SUBSTRATE_RESOLUTION as f32) as usize, ((position.y/768.0)*SUBSTRATE_RESOLUTION as f32) as usize)
}

impl Particle {
    fn update_velocity(&self, tree: &KdTree<Particle>, gt: f32, dt: f32, substrate: &Array2<f32>) -> (Particle, usize) {
        let max_distance = self.family.repel_range + 2.0 * self.family.max_attraction;
        //let max_distance = self.family.max_attraction;
        let push = tree.within_radius(
            &[self.position.x, self.position.y],
            max_distance,
        );

        // let push = tree.nearests(
        //     &[self.position.x, self.position.y],
        //     200,
        // );

        let count = AtomicUsize::new(0);
        let sub_i = get_substrate_index(self.position);
        let phase_offset = substrate[sub_i];
        let reaction_phase = f32::sin(self.family.reaction_freq*gt+self.family.reaction_phase+phase_offset*PI)*0.5+0.5;
        let push = 
            push
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
                    count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let diff = p.position - self.position;
                    let dist = diff.length();
                    let reaction = lerp_tuple(self.family.reactions_a[p.family.id],self.family.reactions_b[p.family.id],reaction_phase);
                    let repel_range = self.family.repel_range;
                    let repel_force =
                        self.family.repel_force + 2.0 * self.fear * self.family.repel_force;
                    let reaction_range = reaction.0;
                    let reaction_force =
                        lerp(reaction.1, -1.0 * reaction.1.abs(), self.fear.powf(4.0));
                    let mult = if dist < self.family.repel_range {
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

        let offset = Vec2 {
            x: 768.0 / 2.0,
            y: 768.0 / 2.0,
        } - self.position;

        let gravity = offset;

        let grav_power = ((gravity.length()/150.0)-1.0).max(0.0);

        let gravity = safe_normalize(gravity) * grav_power.powi(2) * 100.0;

        let menace = if push.1 > 0.0 { push.1 } else { -1.0 };
        let push = push.0;

        (Particle {
            position: self.position,
            velocity: self.velocity
                + (self.velocity * -5.0
                    + (safe_normalize(push) * push.length().min(150.0))
                    + gravity)
                    * dt,
            threat: (self.threat + (menace * 0.0125 - 0.1) * dt).clamp(-1.0, 1.0),
            fear: (self.fear + self.threat * dt).clamp(0.0, 1.0),
            id: self.id,
            family: self.family.clone(),
        },count.load(std::sync::atomic::Ordering::Relaxed))
    }
    fn update_position(self, dt: f32) -> Particle {
        let position = self.position + self.velocity * dt;
        Particle {
            position,
            velocity: self.velocity,
            threat: self.threat,
            fear: self.fear,
            id: self.id,
            family: self.family,
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
    fn new(ctx: &mut Context) -> GameResult<State> {
        let mut particles: Vec<Particle> = vec![];
        let mut rng = rand::thread_rng();
        let mut families: Vec<Family> = Vec::new();

        let p_count = 3_000;

        let family_count = 4;

        let repel_range = 10.0;
        let repel_force = 100.0;
        let min_attact = 2.0;
        let attract_mult = 10.0;
        let max_attraction = repel_range + 2.0 * (min_attact + attract_mult);
        let force = 50.0;


        let h_offset = rng.gen::<f32>() * 360.0;

        for i in 0..family_count {
            let color = Rgb::from_color(Hsv::new(
                (i as f32 / family_count as f32) * 360.0 + h_offset,
                0.6,
                1.0,
            ))
            .into_components();

            families.push(Family {
                color: Color::from(color),
                id: i,
                repel_range,
                repel_force,
                max_attraction,
                reaction_freq: TAU*rng.gen::<f32>()*0.02+0.01,
                reaction_phase: rng.gen::<f32>()*TAU,
                reactions_a: (0..family_count)
                    .map(|_| {
                        (
                            min_attact + rng.gen::<f32>() * attract_mult,
                            rng.gen::<f32>().abs().powf(0.3) * (-1.0 + rng.gen::<f32>()*2.0).signum() * force,
                        )
                    })
                    .collect(),
                reactions_b: (0..family_count)
                    .map(|_| {
                        (
                            min_attact + rng.gen::<f32>() * attract_mult,
                            rng.gen::<f32>().abs().powf(0.3) * (-1.0 + rng.gen::<f32>()*2.0).signum() * force,
                        )
                    })
                    .collect(),
            })
        }

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
                family: Box::from(families[(dist * families.len() as f32) as usize].clone()),
                //family: Box::from(families[rng.gen_range(0..families.len())].clone()),
            })
        }
        Ok(State {
            particles,
            counts: vec![],
            instance_array: InstanceArray::new(ctx, Image::from_path(ctx, "/blob.png")?),
            substrate: Array::zeros((SUBSTRATE_RESOLUTION, SUBSTRATE_RESOLUTION)),
        })
    }
}

impl event::EventHandler<GameError> for State {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        let dt = ctx.time.delta().as_secs_f32();
        let gt = ctx.time.time_since_start().as_secs_f32();

        let tree: KdTree<Particle> = KdTree::par_build_by_ordered_float(self.particles.clone());
        (self.particles, self.counts) = self
            .particles
            .clone()
            .par_iter()
            .map(|p| p.update_velocity(&tree, gt, dt, &self.substrate))
            .map(|p| (p.0.update_position(dt), p.1))
            .collect();

        self.particles.iter().for_each(|particle|{
            let i = get_substrate_index(particle.position);
            self.substrate[i]+=particle.threat.max(particle.fear).max(0.0)*dt;
        });

        let blurred = gaussian_filter(&self.substrate, dt*40.0, 0, BorderMode::Reflect, 3);
        self.substrate = blurred;
        for i in 0..SUBSTRATE_RESOLUTION {
            for j in 0..SUBSTRATE_RESOLUTION {
                self.substrate[(i,j)] = (self.substrate[(i,j)]-dt*0.1).max(0.0).min(1.0);
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
        let mut canvas =
            graphics::Canvas::from_frame(ctx, graphics::Color::from([0.05, 0.025, 0.025, 1.0]));

        self.instance_array.clear();

        canvas.set_blend_mode(graphics::BlendMode::ADD);

        canvas.set_screen_coordinates(Rect { x: 0.0, y: 0.0, w: 768.0, h: 768.0 });

        for particle in self.particles.iter() {
            let i = get_substrate_index(particle.position);
            let sub_val = self.substrate[i];
            let threat = particle.threat * 0.5 + 0.5;
            let activity = threat.max(particle.fear);
            let life = ((particle.velocity.length()-1.0)*0.1).max(0.0).powi(2).clamp(0.0, 1.0);
            let sparkle = ((ctx.time.time_since_start().as_secs_f32() * 5.0)
                + (particle.id as f32 % TAU))
                .sin();
            let sparkle = particle.fear * (1.0 - particle.threat).clamp(0.0, 1.0) * sparkle;
            let sparkle = sparkle.clamp(0.0, 1.0).powf(2.0);
            let l = activity;

            // let color = particle.family.color.to_rgb();
            let color = (
                (particle.fear + sparkle).clamp(0.0, 1.0), //lerp(color.0 as f32 / 256.0, 1.0, threat),
                (threat + sparkle).clamp(0.0, 1.0), //lerp(color.1 as f32 / 256.0, 1.0, threat),
                (1.0 - activity + sparkle).clamp(0.0, 1.0), //lerp(color.2 as f32 / 256.0, 0.0, threat),
            );
            let color = Color::from((
                lerp(particle.family.color.r, color.0, l),
                lerp(particle.family.color.g, color.1, l),
                lerp(particle.family.color.b, color.2, l),
                lerp(0.01, 1.0, life.max(l)),
                //lerp(0.01, 1.0, sub_val),
            ));
            self.instance_array.push(
                DrawParam::new()
                    .offset(Vec2::new(256.0, 256.0))
                    .dest(Point2 {
                        x: particle.position.x,
                        y: particle.position.y,
                    })
                    .color(color)
                    //.rotation(gt*2.0)
                    //.scale(Vec2::new(0.03, 0.03)),
                    .scale(Vec2::new(lerp(0.03,0.06,l.max(1.0-life)), lerp(0.03,0.06,l.max(1.0-life)))),
                    //.scale(Vec2::new(lerp(0.03,0.06,1.0-life), lerp(0.03,0.06,1.0-life))),
                    //.scale(Vec2::new(lerp(0.03,0.06,sub_val), lerp(0.03,0.06,sub_val))),
            );
        }

        canvas.draw(&self.instance_array, DrawParam::default());

        if false {
            let mut fps_text = format!("F/s: {}", ctx.time.fps() as u32);

            let counts: Vec<_> = self.counts.par_iter().map(|count| *count as f32).collect();
            let mean = statistical::mean(&counts);
            let median = statistical::median(&self.counts);
            let mode = statistical::mode(&self.counts).unwrap_or(0);
            let min = *self.counts.par_iter().min().unwrap_or(&0);
            let max = *self.counts.par_iter().max().unwrap_or(&0);

            fps_text.push_str(&format!("\nMin:\t{min}"));
            fps_text.push_str(&format!("\nMax:\t{max}"));
            fps_text.push_str(&format!("\nMean:\t{}", mean as u32));
            fps_text.push_str(&format!("\nMedian:\t{median}"));
            fps_text.push_str(&format!("\nMode:\t{mode}"));

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
        }

        // let sub_divider = 768/SUBSTRATE_RESOLUTION;
        // let sub_scale = [sub_divider as f32, sub_divider as f32];
        // //let sub_scale = [2.0, 2.0];
        // for x in 0..SUBSTRATE_RESOLUTION {
        //     for y in 0..SUBSTRATE_RESOLUTION {
        //         let xx = (x*sub_divider) as f32;
        //         let yy = (y*sub_divider) as f32;
        //         canvas.draw(&Quad, DrawParam::default().color(
        //             Color::from((
        //                 1.0,
        //                 0.0,
        //                 1.0,
        //                 self.substrate[(x,y)],
        //             ))
        //         ).scale(sub_scale).dest([xx, yy]));
        //     }
        // }

        canvas.finish(ctx)?;
        Ok(())
    }
}

pub fn main() {
    let resource_dir = if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        let mut path = path::PathBuf::from(manifest_dir);
        path.push("resources");
        path
    } else {
        path::PathBuf::from("./resources")
    };

    let (mut ctx, event_loop) = ContextBuilder::new("hello_ggez", "awesome_person")
        .window_mode(WindowMode {
            width: 768.0,
            height: 768.0,
            maximized: false,
            fullscreen_type: FullscreenType::Windowed,
            borderless: true,
            resizable: false,
            transparent: false,
            logical_size: Some(LogicalSize::new(768.0, 768.0)),
            ..Default::default()
        })
        .window_setup(WindowSetup {
            title: "Sparkle Particle Life Simulator".to_string(),
            samples: NumSamples::Four,
            vsync: false,
            icon: "".to_string(),
            srgb: false,
        })
        .add_resource_path(resource_dir)
        .backend(Backend::All)
        .build()
        .unwrap();
    let state = State::new(&mut ctx).unwrap();
    event::run(ctx, event_loop, state);
}
