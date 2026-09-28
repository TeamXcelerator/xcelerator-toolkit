use xc_core::{
    run_adaptive_precision, AdaptivePrecisionDecision, AdaptivePrecisionEvaluation,
    PrecisionEscalation, PrecisionPolicy,
};

fn policy(numerator: u32, denominator: u32) -> PrecisionPolicy {
    PrecisionPolicy {
        initial_bits: 64,
        maximum_bits: 256,
        guard_bits: 32,
        escalation: PrecisionEscalation::Multiply {
            numerator,
            denominator,
        },
    }
}

#[test]
fn equivalent_growth_ratios_produce_identical_precision_schedules() {
    let reduced = policy(3, 2);
    let unreduced = policy(3_000_000_000, 2_000_000_000);
    unreduced.validate().unwrap();
    for current in [64, 96, 128, 144, 192, 216, 255, 256] {
        assert_eq!(
            unreduced.next_bits(current),
            reduced.next_bits(current),
            "current={current}"
        );
    }
}

#[test]
fn precision_growth_matches_exact_integer_ceiling_without_premature_saturation() {
    for (numerator, denominator) in [
        (3, 2),
        (u32::MAX, 1),
        (u32::MAX, u32::MAX - 1),
        (3_000_000_000, 2_000_000_000),
        (u32::MAX, 2),
    ] {
        let mut p = policy(numerator, denominator);
        p.maximum_bits = u32::MAX;
        for current in [32, 53, 64, 127, 1024, 1_000_000, u32::MAX / 2, u32::MAX - 1] {
            let product = u128::from(current) * u128::from(numerator);
            let expected = product
                .div_ceil(u128::from(denominator))
                .min(u128::from(u32::MAX)) as u32;
            assert_eq!(
                p.next_bits(current),
                Some(expected),
                "{current} * {numerator}/{denominator}"
            );
            assert!(expected > current);
        }
    }
}

#[test]
fn malformed_precision_growth_is_total_and_cannot_stall() {
    for escalation in [
        PrecisionEscalation::AddBits(0),
        PrecisionEscalation::Multiply {
            numerator: 2,
            denominator: 0,
        },
        PrecisionEscalation::Multiply {
            numerator: 2,
            denominator: 2,
        },
        PrecisionEscalation::Multiply {
            numerator: 1,
            denominator: 2,
        },
    ] {
        let p = PrecisionPolicy {
            escalation,
            ..policy(3, 2)
        };
        let result = std::panic::catch_unwind(|| p.next_bits(64));
        assert!(result.is_ok(), "{escalation:?} panicked");
        assert_eq!(result.unwrap(), None);
    }
}

#[test]
fn adaptive_runner_never_retries_at_lower_or_unchanged_precision() {
    let mut visits = Vec::new();
    let result = run_adaptive_precision(&policy(3_000_000_000, 2_000_000_000), |bits| {
        visits.push(bits);
        let decision = if bits >= 216 {
            AdaptivePrecisionDecision::Accepted {
                diagnostic: "control".into(),
            }
        } else if visits.len() > 8 {
            AdaptivePrecisionDecision::TerminalInconclusive {
                reason: "test guard against a stalled schedule".into(),
            }
        } else {
            AdaptivePrecisionDecision::Retry {
                reason: "more bits required".into(),
            }
        };
        Ok::<_, std::convert::Infallible>(AdaptivePrecisionEvaluation {
            value: bits,
            decision,
        })
    })
    .unwrap();
    assert!(result.is_accepted(), "visits={visits:?}");
    assert_eq!(visits, vec![96, 144, 216]);
}
