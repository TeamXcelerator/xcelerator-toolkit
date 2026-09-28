#![cfg(feature = "arb")]
use rug::{Float, Rational};
use xc_numerics::interval::RationalInterval;
use xc_spectral::ccm::cutoff_free::{assemble, CutoffFreeConfig, CutoffFreeMatrix};

fn rational(s: &str) -> Rational {
    Float::with_val(384, Float::parse(s).unwrap())
        .to_rational()
        .unwrap()
}

#[test]
fn components_agree_with_independent_defining_integrals() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/cutoff_free_oracle.json")).unwrap();
    let slack = rational(oracle["reference_absolute_slack"].as_str().unwrap());
    let mut compared = 0;
    for case in oracle["cases"].as_array().unwrap() {
        let c = case["cutoff"].as_u64().unwrap();
        let modes = case["modes"].as_u64().unwrap() as usize;
        for precision in [64, 128, 192] {
            for short_tail in [true, false] {
                let mut cfg = CutoffFreeConfig::new(c, modes, precision);
                if short_tail {
                    cfg.geometric_terms = 1;
                }
                let matrix = assemble(&cfg).unwrap();
                matrix.validate_assembly().unwrap();
                for (index, entry) in case["entries"].as_array().unwrap().iter().enumerate() {
                    for (component, interval) in [
                        &matrix.w02[index],
                        &matrix.wr[index],
                        &matrix.wp[index],
                        &matrix.tau[index],
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let reference = rational(entry[component].as_str().unwrap());
                        assert!(interval.lower()<=&(reference.clone()+&slack) && interval.upper()>=&(reference-&slack),"C={c} P={precision} short={short_tail} entry={index} component={component}");
                        compared += 1;
                    }
                }
            }
        }
    }
    assert_eq!(compared, 7056);
}

fn assert_rejected(matrix: &CutoffFreeMatrix) {
    assert!(matrix.validate_assembly().is_err());
    assert!(matrix.component_evidence_digest().is_err());
    assert!(matrix.certify_inertia().is_err());
    assert!(matrix.portable_inertia_certificate().is_err());
}

#[test]
fn certificate_requires_the_original_complete_assembly() {
    let original = assemble(&CutoffFreeConfig::new(5, 1, 128)).unwrap();
    original.validate_assembly().unwrap();
    let cert = original.portable_inertia_certificate().unwrap();
    assert_eq!((cert.positive, cert.negative), (3, 0));
    assert!(xc_certify::exact::verify_portable_interval_inertia_certificate(&cert).valid);
    for mutation in 0..13 {
        let mut m = original.clone();
        match mutation {
            0 => {
                m.tau = (0..9)
                    .map(|i| {
                        RationalInterval::point(Rational::from(if i / 3 == i % 3 { -1 } else { 0 }))
                    })
                    .collect()
            }
            1 => {
                m.w02.clear();
                m.wr.clear();
                m.wp.clear();
            }
            2 => m.config.integer_cutoff_c = 1,
            3 => m.config.integer_cutoff_c = 13,
            4 => m.config.modes = usize::MAX,
            5 => m.config.precision_bits = u32::MAX,
            6 => m.config.geometric_terms += 1,
            7 => m.scalar_backend.push_str("-changed"),
            8 => {
                m.w02[0] = RationalInterval::point(Rational::from(1));
                m.tau[0] = m.w02[0].sub(&m.wr[0]).sub(&m.wp[0]);
            }
            9 => {
                m.wr[0] = RationalInterval::point(Rational::from(1));
                m.tau[0] = m.w02[0].sub(&m.wr[0]).sub(&m.wp[0]);
            }
            10 => {
                m.wp[0] = RationalInterval::point(Rational::from(1));
                m.tau[0] = m.w02[0].sub(&m.wr[0]).sub(&m.wp[0]);
            }
            11 => {
                m.tau[0] = RationalInterval::new(
                    m.tau[0].lower().clone() - 1,
                    m.tau[0].upper().clone() + 1,
                )
                .unwrap()
            }
            _ => m.w02[1] = RationalInterval::point(Rational::from(1)),
        }
        assert_rejected(&m);
    }
    assert_eq!(
        original.component_evidence_digest().unwrap(),
        original.clone().component_evidence_digest().unwrap()
    );
}

#[test]
fn invalid_configuration_fails_before_precision_or_dimension_arithmetic() {
    for c in [0, 1, u64::MAX] {
        assert!(assemble(&CutoffFreeConfig::new(c, 0, 64)).is_err());
    }
    for p in [0, 63, 1_000_001, u32::MAX] {
        assert!(assemble(&CutoffFreeConfig::new(5, 0, p)).is_err());
    }
    assert!(assemble(&CutoffFreeConfig::new(5, usize::MAX, 64)).is_err());
    assert!(CutoffFreeConfig::new(5, usize::MAX, 64)
        .checked_dimension()
        .is_err());
    assert_eq!(
        CutoffFreeConfig::new(5, 3, 64).checked_dimension().unwrap(),
        7
    );
    for terms in [0, usize::MAX] {
        let mut cfg = CutoffFreeConfig::new(5, 0, 64);
        cfg.geometric_terms = terms;
        assert!(assemble(&cfg).is_err());
    }
}

#[test]
fn high_precision_inertia_has_bounded_dyadic_pivots_and_portable_replay() {
    // The exact-rational recurrence exceeded a 30-second cap on
    // this 9x9 case as successive Schur denominators grew uncontrollably.
    let matrix = assemble(&CutoffFreeConfig::new(100, 4, 9000)).unwrap();
    let certificate = matrix.portable_inertia_certificate().unwrap();
    assert_eq!(certificate.schema_version, 2);
    assert_eq!(
        (
            certificate.positive,
            certificate.negative,
            certificate.zero_or_unresolved
        ),
        (9, 0, 0)
    );
    assert_eq!(
        certificate.configuration["inertia_semantics"],
        xc_spectral::ccm::cutoff_free::INERTIA_SEMANTICS
    );
    for record in &certificate.pivot_enclosures {
        let interval = xc_certify::exact::parse_interval(record).unwrap();
        for endpoint in [interval.lower(), interval.upper()] {
            assert!(endpoint.numer().significant_bits() < 10000);
            assert!(endpoint.denom().significant_bits() < 10000);
        }
    }
    assert!(xc_certify::exact::verify_portable_interval_inertia_certificate(&certificate).valid);
}
