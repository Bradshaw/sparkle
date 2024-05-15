use ggez::glam::Vec2;

pub(crate) fn inflect(x: f32, curve: f32) -> f32 {
    let ease_out = |v: f32| 1.0 - 2.0f32.powf(-curve * v);
    let a = ease_out(x);
    let b = x * (1.0 - ease_out(1.0));
    a + b
}

pub(crate) fn get_downsample_position(position: Vec2, resolution: (usize, usize)) -> Vec2 {
    (
        (position.x / 768.0) * resolution.0 as f32,
        (position.y / 768.0) * resolution.1 as f32,
    )
        .into()
}

pub(crate) fn bilerp(x1y1: f32, x2y1: f32, x1y2: f32, x2y2: f32, tx: f32, ty: f32) -> f32 {
    lerp(lerp(x1y1, x2y1, tx), lerp(x1y2, x2y2, tx), ty)
}

pub(crate) fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    //x*x*(3.0-2.0*x)
    (x * x * x * (x * (6.0 * x - 15.0) + 10.0)).clamp(0.0, 1.0)
}

pub(crate) fn smoothtable(x: f32, rise: f32, fall: f32) -> f32 {
    smoothstep((x * rise).clamp(0.0, 1.0) * (fall - x * fall).clamp(0.0, 1.0))
}

pub(crate) fn safe_normalize(v: Vec2) -> Vec2 {
    if v.length_squared() < f32::EPSILON {
        v
    } else {
        v.normalize()
    }
}

pub(crate) fn lerp(a: f32, b: f32, n: f32) -> f32 {
    b * n + a * (1.0 - n)
}

pub(crate) fn lerp_tuple(a: (f32, f32), b: (f32, f32), n: f32) -> (f32, f32) {
    (lerp(a.0, b.0, n), lerp(a.1, b.1, n))
}
