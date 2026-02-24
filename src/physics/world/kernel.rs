use bevy::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridKernelSample {
    pub node: IVec2,
    pub weight: f32,
    pub grad: Vec2,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GridKernelStencil {
    samples: Vec<GridKernelSample>,
}

impl GridKernelStencil {
    pub fn samples(&self) -> &[GridKernelSample] {
        &self.samples
    }
}

pub fn evaluate_quadratic_bspline_stencil_2d(position: Vec2, inv_h: f32) -> GridKernelStencil {
    let grid_pos = position * inv_h;
    let base = (grid_pos - Vec2::splat(0.5)).floor().as_ivec2();
    let mut samples = Vec::with_capacity(9);
    for oy in 0..3 {
        for ox in 0..3 {
            let node = base + IVec2::new(ox, oy);
            let rel = grid_pos - node.as_vec2();
            let (wx, dwx) = quadratic_bspline_weight_and_grad_1d(rel.x);
            let (wy, dwy) = quadratic_bspline_weight_and_grad_1d(rel.y);
            let weight = wx * wy;
            let grad = Vec2::new(dwx * wy, wx * dwy) * inv_h;
            samples.push(GridKernelSample { node, weight, grad });
        }
    }
    GridKernelStencil { samples }
}

fn quadratic_bspline_weight_and_grad_1d(distance: f32) -> (f32, f32) {
    let abs_dist = distance.abs();
    if abs_dist < 0.5 {
        let w = 0.75 - abs_dist * abs_dist;
        let grad = -2.0 * distance;
        (w, grad)
    } else if abs_dist < 1.5 {
        let t = 1.5 - abs_dist;
        let w = 0.5 * t * t;
        let grad = -distance.signum() * t;
        (w, grad)
    } else {
        (0.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_bspline_weights_sum_to_one() {
        let stencil = evaluate_quadratic_bspline_stencil_2d(Vec2::new(2.25, -0.125), 4.0);
        let weight_sum: f32 = stencil.samples().iter().map(|sample| sample.weight).sum();
        assert!((weight_sum - 1.0).abs() <= 1e-5);
        assert_eq!(stencil.samples().len(), 9);
    }

    #[test]
    fn quadratic_bspline_gradients_conserve_constant_field() {
        let stencil = evaluate_quadratic_bspline_stencil_2d(Vec2::new(0.375, 0.625), 4.0);
        let grad_sum = stencil
            .samples()
            .iter()
            .fold(Vec2::ZERO, |acc, sample| acc + sample.grad);
        assert!(grad_sum.length() <= 1e-5);
    }

    #[test]
    fn stencil_node_order_is_stable_for_fixed_position() {
        let a = evaluate_quadratic_bspline_stencil_2d(Vec2::new(0.5, 0.5), 2.0);
        let b = evaluate_quadratic_bspline_stencil_2d(Vec2::new(0.5, 0.5), 2.0);
        assert_eq!(a, b);
    }
}
