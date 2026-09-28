#![cfg(feature = "hp")]
use xc_variational::maynard::*;
#[test]
fn partition_and_projector_admission_precedes_excessive_exact_work() {
    assert!(matches!(
        enumerate_integer_partitions(5, 1000),
        Err(MkError::InvalidProblem(_))
    ));
    assert!(matches!(
        enumerate_integer_partitions(50, 200),
        Err(MkError::InvalidProblem(_))
    ));
    assert_eq!(enumerate_integer_partitions(5, 20).unwrap().len(), 1125);
    // Every degree in one variable has one partition; this also exercises
    // pruning of impossible recursion branches in a long thin space.
    assert_eq!(enumerate_integer_partitions(1, 1000).unwrap().len(), 1001);
    let large = MkMonomialReference::new(8, 8).unwrap();
    assert!(matches!(
        MkSectorProjector::new(&large, MkPermutationSector::Trivial),
        Err(MkError::InvalidProblem(_))
    ));
    let policy = AdaptiveSpacePolicy {
        k: 5,
        initial_degree: 0,
        maximum_degree: 80,
        maximum_generations: 81,
        enrichment_rule: AdaptiveEnrichmentRule::CompleteDegreeShell,
    };
    assert!(matches!(
        build_adaptive_symmetric_spaces(&policy),
        Err(MkError::InvalidProblem(_))
    ));
}
#[test]
fn sector_aliases_identify_the_same_exact_representation() {
    let reference = MkMonomialReference::new(3, 2).unwrap();
    let (_, report) = exact_sector_coverage(&reference).unwrap();
    for (named, shape) in [
        (MkPermutationSector::Trivial, vec![3]),
        (MkPermutationSector::Standard, vec![2, 1]),
        (MkPermutationSector::Alternating, vec![1, 1, 1]),
    ] {
        assert_eq!(
            report.sector_dimension(&named),
            report.sector_dimension(&MkPermutationSector::Partition(IntegerPartition(shape)))
        );
    }
    assert_eq!(
        report.sector_dimension(&MkPermutationSector::Trivial),
        Some(4)
    );
}
#[test]
fn degree_band_certificate_is_bound_to_its_declared_form_and_basis() {
    let reference = MkSymmetricReference::new(3, 4).unwrap();
    let metric =
        MkCertifiedDegreeBandAction::construct(&reference, MkSymmetricForm::IMetric, 1).unwrap();
    let total =
        MkCertifiedDegreeBandAction::construct(&reference, MkSymmetricForm::JTotal, 1).unwrap();
    for action in [&metric, &total] {
        validate_mk_acceleration(&action.acceleration(), MkAssuranceMode::Certified, 3, 4).unwrap();
    }
    let wrong = MkOperatorAcceleration::DegreeDifferenceBand {
        form: MkSymmetricForm::JTotal,
        basis: MkAccelerationBasis::SymmetricOrbitSum,
        half_width: 1,
        certificate: Some(metric.certificate.clone()),
    };
    assert!(validate_mk_acceleration(&wrong, MkAssuranceMode::Certified, 3, 4).is_err());
    // Removing the typed form/basis cannot recover the old ambiguous descriptor.
    let mut encoded = serde_json::to_value(total.acceleration()).unwrap();
    encoded.as_object_mut().unwrap().remove("form");
    assert!(serde_json::from_value::<MkOperatorAcceleration>(encoded).is_err());
}
