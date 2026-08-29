// Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.®)
// All rights reserved. See LICENSE in the repository root.

//! Cross-check card for a runtime-supplied target and weighted distances.
//!
//! Prints the reference constants and a rule-sensitivity table so that
//! independent implementations can be compared line by line. No integration
//! rule here is authoritative; the point of the table is that a reported
//! distance is meaningless without the rule that produced it.
//!
//! Run: `cargo run -p xc-spectral --example target_distance`

use xc_numerics::grid_integral::{GridVariable, UniformGridScheme};
use xc_spectral::distance::{
    distance_to_target_f64, weighted_alpha_norm_f64, WeightedIntegrationRule,
};
use xc_spectral::target::TargetEvaluatorF64;

fn main() -> anyhow::Result<()> {
    let target = TargetEvaluatorF64::from_environment()?;
    println!("Runtime target cross-check card (binary64 tier)");
    println!("====================================================");
    println!("definition digest: {}", target.definition_digest());
    println!("normalized target values:");
    for u in [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0] {
        println!("  target({u:>4}) = {:.16e}", target.value(u));
    }
    println!();

    // Weighted norm of the supplied target itself over [1, 20].
    let target_norm = weighted_alpha_norm_f64(
        |u| target.value(u),
        20.0,
        0.5,
        WeightedIntegrationRule::UniformGrid {
            scheme: UniformGridScheme::Trapezoid,
            variable: GridVariable::U,
            steps: 400_000,
        },
    )?;
    println!(
        "int_1^20 target(u) u^(-1/2) du = {:.13}   [{}, {}, resolution={}]",
        target_norm.value,
        target_norm.rule.rule(),
        target_norm.rule.variable().as_str(),
        target_norm.rule.resolution(),
    );
    println!();

    // Rule sensitivity. The profile f = 1 stands in for a normalized
    // eigenfunction; every rule below approximates the same finite integral,
    // so their spread exposes quadrature sensitivity.
    let lambda = 17.0_f64.sqrt();
    let resolution = ((lambda - 1.0) * 1000.0).ceil() as usize;
    println!("d(f=1, lambda=sqrt(17)), alpha = 1/2, comparable resolutions:");
    let mut rules: Vec<WeightedIntegrationRule> = Vec::new();
    for scheme in [
        UniformGridScheme::LeftRiemann,
        UniformGridScheme::RightRiemann,
        UniformGridScheme::Midpoint,
        UniformGridScheme::Trapezoid,
    ] {
        rules.push(WeightedIntegrationRule::UniformGrid {
            scheme,
            variable: GridVariable::U,
            steps: resolution,
        });
    }
    for points in [200, 600, 1200] {
        rules.push(WeightedIntegrationRule::GaussLegendre {
            points,
            variable: GridVariable::U,
        });
    }
    for rule in rules {
        let result = distance_to_target_f64(|_| 1.0, lambda, 0.5, rule)?;
        println!(
            "  {:<14} res={:<6} -> {:.12}",
            result.rule.rule(),
            result.rule.resolution(),
            result.value,
        );
    }
    println!();
    println!("For this smooth integrand Gauss-Legendre converges far faster than any");
    println!("uniform rule. That advantage is not guaranteed for a real eigenfunction:");
    println!("an absolute residual has a derivative kink at every interior sign change,");
    println!("where Gauss-Legendre drops to algebraic convergence. Report the rule with");
    println!("the number, and choose it from the integrand rather than by convention.");
    Ok(())
}
