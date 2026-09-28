//! Independent derivative-integral oracle for exact retained quadrature points.
use super::*;

#[test]
fn quadrant_budget_admits_ordinary_high_precision_rule() {
    // HP-1000 default shape. These exact dyadic points exercise only the
    // per-node conversion budget; no expensive quadrature generation is needed.
    let nodes = (0..3000i32)
        .map(|index| Float::with_val(3386, index - 1500) / 2048u32)
        .collect::<Vec<_>>();
    assert!(
        crate::ccm::certified_roots::boundary::rational_point_vector_budget(nodes.iter()).is_ok()
    );
    for node in &nodes {
        assert!(exact_quadrant(node, 2).is_ok());
    }
    let compact_extreme = Float::with_val(3386, 1) >> 1_000_000_000u32;
    assert!(exact_quadrant(&compact_extreme, 2).is_err());
    assert!(
        crate::ccm::certified_roots::boundary::rational_point_vector_budget(std::iter::once(
            &compact_extreme
        ))
        .is_err()
    );
    assert_eq!(
        exact_quadrant(&Float::with_val(3386, 0), 1).unwrap(),
        Some(2)
    );
}

fn alpha_oracle(mode: usize, length: &Float, nodes: &[Float], weights: &[Float]) -> I {
    let p = 2048;
    let one = I::from_i64(1, p);
    let two = I::from_i64(2, p);
    let length = I::from_float(length, p).unwrap();
    let mut total = I::from_i64(0, p);
    for (node, weight) in nodes.iter().zip(weights) {
        let h = I::from_float(node, p).unwrap().add(&one).div(&two).unwrap();
        let x = length.mul(&h);
        let ep = x.exp();
        let em = x.neg().exp();
        let rho = x.div(&two).unwrap().exp().div(&ep.sub(&em)).unwrap();
        let coth = ep.add(&em).div(&ep.sub(&em)).unwrap();
        let derivative = rho.mul(&one.div(&two).unwrap().sub(&coth));
        let phase = I::pi(p).mul(&I::from_u64((2 * mode) as u64, p)).mul(&h);
        let term = I::from_float(weight, p)
            .unwrap()
            .mul(&phase.sin())
            .mul(&rho.add(&x.mul(&derivative)));
        total = total.add(&term);
    }
    total.div(&I::pi(p).mul(&two)).unwrap()
}

#[test]
fn actual_generated_center_must_not_become_an_exact_quadrant() {
    let source_p = 64;
    let work = source_p + 64;
    let (nodes, weights) = xc_numerics::quadrature::try_gauss_legendre_nodes(
        7,
        source_p,
        xc_numerics::quadrature::CacheMode::Off,
    )
    .unwrap();
    let node = &nodes[3];
    assert_ne!(node, &0);
    let length = Float::with_val(source_p, 2);
    // This admitted one-node rule isolates the contribution of an actual GL
    // center, without allowing unrelated wide terms to hide its enclosure loss.
    let nodes = std::slice::from_ref(node);
    let weights = std::slice::from_ref(&weights[3]);
    let actual = integral(
        1,
        &I::from_float(&length, work).unwrap(),
        nodes,
        weights,
        work,
    )
    .unwrap();
    let oracle = alpha_oracle(1, &length, nodes, weights);
    println!(
        "retained center={node}, alpha interval=[{},{}], independent=[{},{}]",
        actual[0].lower(),
        actual[0].upper(),
        oracle.lower(),
        oracle.upper()
    );
    assert!(oracle.upper() < &0);
    assert!(actual[0].lower() <= oracle.lower() && actual[0].upper() >= oracle.upper());
}

#[test]
fn complete_generated_rules_compare_to_independent_derivative_integral() {
    for (source_p, order) in [
        (64, 35),
        (64, 37),
        (128, 67),
        (128, 69),
        (257, 131),
        (257, 133),
    ] {
        let work = source_p + 64;
        let (nodes, weights) = xc_numerics::quadrature::try_gauss_legendre_nodes(
            order,
            source_p,
            xc_numerics::quadrature::CacheMode::Off,
        )
        .unwrap();
        let length = Float::with_val(source_p, 2);
        let actual = integral(
            1,
            &I::from_float(&length, work).unwrap(),
            &nodes,
            &weights,
            work,
        )
        .unwrap();
        let oracle = alpha_oracle(1, &length, &nodes, &weights);
        let contains = actual[0].lower() <= oracle.lower() && actual[0].upper() >= oracle.upper();
        println!(
            "full GL p={source_p}, order={order}, center={}, contains_oracle={contains}",
            nodes[order / 2]
        );
        assert!(contains);
    }
}
