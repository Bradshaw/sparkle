use clap::{Parser, ValueEnum};
use rand::{rngs::SmallRng, Rng, SeedableRng};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub(crate) enum ExplicitBoolean {
    Yes,
    No,
}

impl From<ExplicitBoolean> for bool {
    fn from(value: ExplicitBoolean) -> Self {
        value == ExplicitBoolean::Yes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub(crate) enum ConnectingLines {
    None,
    Family,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub(crate) enum ScreenMode {
    Window,
    BorderlessWindow,
    Fullscreen,
    DesktopFullscreen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub(crate) enum SimulationCatchup {
    Slowdown,
    IgnoreUpdateRate,
}

/// Sparkle
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub(crate) struct Args {
    /// Seed to use
    #[arg(short, long)]
    pub(crate) seed: Option<u64>,

    /// Number of particles
    #[arg(short, long, default_value = "3000")]
    pub(crate) particle_count: u16,

    /// Number of families
    #[arg(short, long, default_value = "4")]
    pub(crate) family_count: u8,

    /// Simulation update rate
    #[arg(short, long, default_value = "30")]
    pub(crate) update_rate: f32,

    /// Limit rendering to display refresh rate
    #[arg(short, long, default_value = "yes")]
    pub(crate) vsync: ExplicitBoolean,

    /// Catchup strategy when rendering is slower than simulation
    #[arg(long, default_value = "slowdown")]
    pub(crate) catchup: SimulationCatchup,

    /// How to draw connecting lines
    #[arg(short, long, value_enum, default_value = "family")]
    pub(crate) connecting_lines: ConnectingLines,

    /// Length of connecting lines
    #[arg(short = 'l', long, default_value = "10")]
    pub(crate) connecting_line_length: f32,

    /// Display mode to use
    #[arg(short = 'm', long, value_enum, default_value = "window")]
    pub(crate) screen_mode: ScreenMode,

    /// Resolution
    #[arg(short, long, default_value = "600")]
    pub(crate) resolution: u16,

    /// In desktop and windowed modes, use a transparent background
    #[arg(short, long, default_value = "no")]
    pub(crate) transparent: ExplicitBoolean,
}

pub(crate) enum Seed {
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
    pub(crate) fn get_seed(&self) -> u64 {
        match self {
            Seed::Entropy => {
                let mut rng = SmallRng::from_entropy();
                rng.gen()
            }
            Seed::U64(u) => *u,
        }
    }
}
