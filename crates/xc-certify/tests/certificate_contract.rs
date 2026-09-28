use xc_certify::{DecimalInterval, EigenvalueEnclosure, SpectralGapCertificate};
fn cluster(index: usize, lower: &str, upper: &str) -> EigenvalueEnclosure {
    EigenvalueEnclosure {
        first_index: index,
        last_index: index,
        interval: DecimalInterval {
            lower: lower.into(),
            upper: upper.into(),
        },
        multiplicity_lower: 1,
        multiplicity_upper: 1,
    }
}
#[test]
fn a_gap_bound_must_not_exceed_the_proved_endpoint_separation() {
    let mut gap = SpectralGapCertificate {
        lower_cluster: cluster(0, "0", "1"),
        upper_cluster: cluster(1, "2", "3"),
        certified_lower_bound: "100".into(),
    };
    assert!(gap.validate().is_err());
    gap.certified_lower_bound = "1".into();
    assert!(gap.validate().is_ok());
    gap.certified_lower_bound = "1.0000000000000000000000000000000000000001".into();
    assert!(gap.validate().is_err());
}
#[test]
fn a_gap_requires_adjacent_clusters_with_consistent_multiplicity() {
    // Eigenvalues 1..=4 would lie between clusters indexed 0 and 5.
    let mut gap = SpectralGapCertificate {
        lower_cluster: cluster(0, "0", "1"),
        upper_cluster: cluster(5, "2", "3"),
        certified_lower_bound: "1".into(),
    };
    assert!(gap.validate().is_err());
    gap.upper_cluster = cluster(1, "2", "3");
    assert!(gap.validate().is_ok());
    let mut oversized = cluster(0, "0", "1");
    oversized.last_index = usize::MAX;
    assert!(oversized.validate().is_err());
    // One indexed eigenvalue cannot have multiplicity three.
    gap.lower_cluster.multiplicity_lower = 3;
    gap.lower_cluster.multiplicity_upper = 3;
    assert!(gap.validate().is_err());
    // A two-eigenvalue cluster is consistent with bounds [1, 2].
    gap.lower_cluster = cluster(0, "0", "1");
    gap.lower_cluster.last_index = 1;
    gap.lower_cluster.multiplicity_upper = 2;
    gap.upper_cluster = cluster(2, "2", "3");
    assert!(gap.validate().is_ok());
}
#[test]
fn reversing_clusters_is_not_a_positive_gap_certificate() {
    let gap = SpectralGapCertificate {
        lower_cluster: cluster(0, "2", "3"),
        upper_cluster: cluster(1, "0", "1"),
        certified_lower_bound: "0.1".into(),
    };
    assert!(gap.validate().is_err());
}
