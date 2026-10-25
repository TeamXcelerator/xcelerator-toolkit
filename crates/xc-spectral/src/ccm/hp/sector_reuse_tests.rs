//! Frozen standalone producer versus admitted-bounds/shared-guide qualification.
use super::*;

struct Fixture {
    params: CcmParams,
    cfg: HighPrecConfig,
    matrix: Vec<Float>,
    tri: SectorTridiagonalHp,
    transform: SectorTransformHp,
    values: SectorEigenvaluesHp,
    bounds: Vec<(Float, Float)>,
    allowance: Float,
    parity: CcmParity,
    count: usize,
}
impl Fixture {
    fn new(n: usize, p: u32, parity: CcmParity, route: CcmSectorEigenvalueRoute) -> Self {
        let params = CcmParams::from_lambda_sq_integer(
            13,
            match parity {
                CcmParity::Even => n - 1,
                CcmParity::Odd => n,
            },
        );
        let mut cfg = HighPrecConfig::for_decimal_digits(30);
        cfg.precision_bits = p;
        cfg.inverse_iter_steps = 24;
        let mut matrix = vec![Float::with_val(p, 0); n * n];
        for i in 0..n {
            matrix[i * n + i] = Float::with_val(p, 3 * i + 4);
            if i + 1 < n {
                matrix[i * n + i + 1] = Float::with_val(p, 0.125);
                matrix[(i + 1) * n + i] = Float::with_val(p, 0.125);
            }
        }
        let (diagonal, off_diagonal, basis) =
            xc_numerics::eigen::householder_tridiag_hp_stable(&matrix, n, p).unwrap();
        let tri = SectorTridiagonalHp {
            diagonal,
            off_diagonal,
        };
        let transform = SectorTransformHp { basis };
        let count = n.min(8);
        let values = compute_sector_eigenvalues(&tri, n, count, route, p).unwrap();
        let allowance = sector_transform_validation::bounds(
            &matrix,
            &tri.diagonal,
            &tri.off_diagonal,
            &transform.basis,
            n,
            p,
        )
        .unwrap()
        .eigenvalue_allowance;
        let bounds =
            sector_resolution::bounds_with_allowance(&allowance, &tri, &values, count, p).unwrap();
        Self {
            params,
            cfg,
            matrix,
            tri,
            transform,
            values,
            bounds,
            allowance,
            parity,
            count,
        }
    }
    fn reference(&self) -> Result<CcmSectorSpectrumHp> {
        compute_sector_spectrum(
            &self.matrix,
            self.tri.diagonal.len(),
            self.parity,
            self.count,
            &self.values,
            &self.tri,
            &self.transform,
            &self.cfg,
        )
    }
    fn optimized(&self) -> Result<CcmSectorSpectrumHp> {
        compute_sector_spectrum_with_bounds(
            &self.matrix,
            self.tri.diagonal.len(),
            self.parity,
            self.count,
            &self.values,
            &self.tri,
            &self.transform,
            &self.cfg,
            &self.bounds,
            true,
        )
    }
    fn portable(&self, spectrum: &CcmSectorSpectrumHp) -> PortableSectorSpectrum {
        let mut value = portable_sector_spectrum(spectrum);
        value.lambda_squared = lambda_squared_cache_identity(&self.params);
        value.n_modes = self.params.n_modes;
        value.precision_bits = self.cfg.precision_bits;
        value
    }
    fn decode(
        &self,
        portable: &PortableSectorSpectrum,
    ) -> std::result::Result<CcmSectorSpectrumHp, CacheError> {
        decode_sector_spectrum(
            portable,
            &self.params,
            &self.cfg,
            self.parity,
            self.values.route,
            &self.matrix,
            self.tri.diagonal.len(),
            self.count,
            &self.values.values,
            &self.bounds,
        )
    }
}

fn exact_scalars(s: &CcmSectorSpectrumHp) -> Vec<(u32, bool, String)> {
    s.eigenpairs
        .iter()
        .flat_map(|pair| {
            pair.eigenvector.iter().chain([
                &pair.eigenvalue,
                &pair.residual_norm,
                &pair.eigenvalue_lower,
                &pair.eigenvalue_upper,
            ])
        })
        .map(|x| (x.prec(), x.is_sign_negative(), x.to_string_radix(16, None)))
        .collect()
}

#[test]
fn sector_reuse_all_routes_and_parities_preserve_complete_payload_bytes() {
    for parity in [CcmParity::Even, CcmParity::Odd] {
        for route in [
            CcmSectorEigenvalueRoute::Selected,
            CcmSectorEigenvalueRoute::CompleteQr,
            CcmSectorEigenvalueRoute::CrossChecked,
        ] {
            let f = Fixture::new(6, 128, parity, route);
            let full_bounds = sector_resolution::bounds(
                &f.matrix,
                &f.tri,
                &f.transform,
                &f.values,
                f.count,
                f.cfg.precision_bits,
            )
            .unwrap();
            for (left, right) in full_bounds.iter().zip(&f.bounds) {
                for (a, b) in [(&left.0, &right.0), (&left.1, &right.1)] {
                    assert_eq!(a.prec(), b.prec());
                    assert_eq!(a.total_cmp(b), std::cmp::Ordering::Equal);
                }
            }
            let old = f.reference().unwrap();
            SECTOR_SHARED_GUIDES.with(|count| count.set(0));
            let new = f.optimized().unwrap();
            assert_eq!(SECTOR_SHARED_GUIDES.with(std::cell::Cell::get), 1);
            assert_eq!(exact_scalars(&new), exact_scalars(&old));
            assert_eq!(
                serde_json::to_vec(&f.portable(&new)).unwrap(),
                serde_json::to_vec(&f.portable(&old)).unwrap()
            );
            let portable = f.portable(&new);
            assert!(f.decode(&portable).is_ok());
            let mut changed = portable.clone();
            changed.eigenvalue_bounds[0].0 = "0".into();
            assert!(f.decode(&changed).is_err());
            let mut changed = portable.clone();
            changed.eigenvectors[0][0] = "2".into();
            assert!(f.decode(&changed).is_err());
            let mut changed = portable;
            changed.n_modes += 1;
            assert!(f.decode(&changed).is_err());
        }
    }
}

#[test]
fn sector_reuse_retains_exact_ambiguity_and_request_refusals() {
    for parity in [CcmParity::Even, CcmParity::Odd] {
        let mut f = Fixture::new(3, 128, parity, CcmSectorEigenvalueRoute::CompleteQr);
        let p = f.cfg.precision_bits;
        let tiny = Float::with_val(p, 2).pow(-126i32);
        f.matrix = vec![Float::with_val(p, 0); 9];
        f.matrix[4] = tiny.clone();
        f.matrix[8] = Float::with_val(p, 1);
        f.tri.diagonal = vec![Float::with_val(p, 0), tiny, Float::with_val(p, 1)];
        f.tri.off_diagonal = vec![Float::with_val(p, 0); 2];
        f.transform.basis = (0..9)
            .map(|i| Float::with_val(p, u32::from(i / 3 == i % 3)))
            .collect();
        f.values.values = f.tri.diagonal.clone();
        f.count = 2;
        f.bounds =
            sector_resolution::bounds(&f.matrix, &f.tri, &f.transform, &f.values, 2, p).unwrap();
        let old = f.reference().unwrap_err();
        SECTOR_SHARED_GUIDES.with(|count| count.set(0));
        let new = f.optimized().unwrap_err();
        assert_eq!(old.to_string(), new.to_string());
        assert!(new.to_string().contains("2 candidate members"));
        assert_eq!(SECTOR_SHARED_GUIDES.with(std::cell::Cell::get), 1);
        assert_eq!(
            old.downcast_ref::<xc_numerics::eigen::HpEigenvectorRecoveryFailure>(),
            new.downcast_ref::<xc_numerics::eigen::HpEigenvectorRecoveryFailure>()
        );
    }
    let mut f = Fixture::new(3, 128, CcmParity::Even, CcmSectorEigenvalueRoute::Selected);
    f.cfg.inverse_iter_steps = 0;
    assert_eq!(
        f.reference().unwrap_err().to_string(),
        f.optimized().unwrap_err().to_string()
    );
    f.cfg.inverse_iter_steps = 24;
    f.count = 1;
    f.bounds.truncate(1);
    f.values.values[0] += 100;
    assert_eq!(
        f.reference().unwrap_err().to_string(),
        f.optimized().unwrap_err().to_string()
    );
    f.count = 0;
    assert_eq!(
        f.reference().unwrap_err().to_string(),
        f.optimized().unwrap_err().to_string()
    );
}

#[test]
fn sector_reuse_changed_exponent_range_keeps_the_original_recovery_route() {
    unsafe extern "C" {
        fn mpfr_set_emin(value: std::os::raw::c_long) -> i32;
    }
    struct Restore(i32);
    impl Drop for Restore {
        fn drop(&mut self) {
            assert_eq!(unsafe { mpfr_set_emin(self.0 as _) }, 0);
        }
    }
    let f = Fixture::new(3, 128, CcmParity::Even, CcmSectorEigenvalueRoute::Selected);
    let prepared = xc_numerics::eigen::TridiagonalEigenvectorRecoveryHp::new(
        &f.tri.diagonal,
        &f.tri.off_diagonal,
        f.cfg.precision_bits,
        xc_numerics::eigen::TridiagEigvecOptions {
            max_steps: f.cfg.inverse_iter_steps,
            early_termination: true,
            solver: xc_numerics::eigen::TridiagSolver::BandedInterleaved,
        },
    )
    .unwrap()
    .prepare()
    .unwrap();
    let restore = Restore(rug::float::exp_min());
    assert_eq!(unsafe { mpfr_set_emin(-1000) }, 0);
    assert!(!prepared.matches_current_exponent_range());
    let zero = Float::with_val(f.cfg.precision_bits, 0);
    for value in &f.values.values {
        let old = recover_sector_tridiagonal_vector(None, &f.tri, value, &zero, &f.cfg);
        let new = recover_sector_tridiagonal_vector(Some(&prepared), &f.tri, value, &zero, &f.cfg);
        match (old, new) {
            (Ok(old), Ok(new)) => {
                for (a, b) in old.iter().zip(&new) {
                    assert_eq!(a.prec(), b.prec());
                    assert_eq!(a.total_cmp(b), std::cmp::Ordering::Equal);
                }
            }
            (Err(old), Err(new)) => assert_eq!(old.to_string(), new.to_string()),
            (old, new) => panic!("exponent-route result changed: {old:?} {new:?}"),
        }
    }
    drop(restore);
    assert!(prepared.matches_current_exponent_range());
}

#[test]
fn sector_reuse_managed_cold_warm_keys_dependencies_and_bytes_match_reference() {
    managed_cold_warm(6, 192);
}

#[test]
#[ignore = "explicit bounded timing fixture, not a full claim benchmark"]
fn sector_reuse_managed_timing_fixture() {
    managed_cold_warm(64, 256);
}

fn managed_cold_warm(dimension: usize, precision: u32) {
    use xc_cache::{
        ArtifactExecutionCacheMode, CacheLayer, CachePolicy, CacheResolver, CacheVisibility,
        FilesystemCacheStore,
    };
    let root = xc_core::test_support::TestDir::new("sector-reuse-equivalence");
    let policy = CachePolicy {
        current_toolkit_version: ToolkitVersion::parse(env!("CARGO_PKG_VERSION")).unwrap(),
        minimum_quality: CacheQuality::Validated,
        accepted_schema_versions: vec![1],
        allow_deprecated: false,
        allow_quarantined: false,
        allowed_visibilities: vec![CacheVisibility::Local],
    };
    for parity in [CcmParity::Even, CcmParity::Odd] {
        for route in [
            CcmSectorEigenvalueRoute::Selected,
            CcmSectorEigenvalueRoute::CompleteQr,
            CcmSectorEigenvalueRoute::CrossChecked,
        ] {
            let f = Fixture::new(dimension, precision, parity, route);
            let parents = ["matrix", "values", "tridiagonal", "transform"].map(|label| {
                super::tests::conditioning_test_manifest(&format!("ccm_sector_{label}"), label)
            });
            let run = |reference: bool| {
                let dir = root.to_path_buf().join(format!(
                    "{}-{}-{reference}",
                    parity.as_str(),
                    route.as_str()
                ));
                let resolver = CacheResolver::new(vec![CacheLayer {
                    precedence: 0,
                    store: Box::new(FilesystemCacheStore::new(
                        "workstation",
                        dir,
                        true,
                        CacheVisibility::Local,
                    )),
                }]);
                let context = ArtifactCacheContext {
                    resolver: Some(&resolver),
                    reference_resolver: None,
                    acceptance: Some(&policy),
                    ordered_overlays: vec!["workstation".into()],
                    mode: ArtifactExecutionCacheMode::PreferReuse,
                    write_on_miss: true,
                    write_visibility: CacheVisibility::Local,
                    requested_assurance: xc_core::AssuranceLevel::Computed,
                    certification_failure_policy:
                        xc_cache::CertificationFailurePolicy::RetainComputedFailRun,
                    production_sink: None,
                };
                SECTOR_REFERENCE_PRODUCER.with(|v| v.set(reference));
                let resolve = || {
                    resolve_sector_spectrum_via_cache(
                        &f.params,
                        &f.cfg,
                        parity,
                        &f.matrix,
                        &parents[0],
                        f.count,
                        &f.values,
                        &parents[1],
                        &f.tri,
                        &parents[2],
                        &f.transform,
                        &parents[3],
                        &f.allowance,
                        &context,
                    )
                    .unwrap()
                };
                SECTOR_SHARED_GUIDES.with(|count| count.set(0));
                let start = Instant::now();
                let cold = resolve();
                let cold_seconds = start.elapsed().as_secs_f64();
                assert_eq!(
                    SECTOR_SHARED_GUIDES.with(std::cell::Cell::get),
                    usize::from(!reference)
                );
                SECTOR_SHARED_GUIDES.with(|count| count.set(0));
                let start = Instant::now();
                let warm = resolve();
                let warm_seconds = start.elapsed().as_secs_f64();
                assert_eq!(SECTOR_SHARED_GUIDES.with(std::cell::Cell::get), 0);
                SECTOR_REFERENCE_PRODUCER.with(|v| v.set(false));
                assert_eq!(cold.1.key, warm.1.key);
                assert_eq!(cold.1.content_digest, warm.1.content_digest);
                assert_eq!(cold.1.dependencies, warm.1.dependencies);
                assert_eq!(exact_scalars(&cold.0), exact_scalars(&warm.0));
                let bytes = resolver
                    .read_exact_payload(&cold.1.key, &cold.1.content_digest, &policy)
                    .unwrap();
                eprintln!("sector reuse timing: parity={} route={} reference={} dimension={dimension} precision={precision} cold_seconds={cold_seconds:.6} warm_seconds={warm_seconds:.6}",parity.as_str(),route.as_str(),reference);
                (cold.1, bytes)
            };
            let old = run(true);
            let new = run(false);
            assert_eq!(old.0.key, new.0.key);
            assert_eq!(old.0.dependencies, new.0.dependencies);
            assert_eq!(old.0.content_digest, new.0.content_digest);
            assert_eq!(old.1, new.1);
            assert_eq!(
                new.0.dependencies,
                canonical_dependency_refs(parents.to_vec())
            );
        }
    }
}

#[test]
fn sector_reuse_dense_transform_roundtrip_preserves_proof_payload_and_refusal() {
    for p in [64, 512] {
        for parity in [CcmParity::Even, CcmParity::Odd] {
            for route in [
                CcmSectorEigenvalueRoute::Selected,
                CcmSectorEigenvalueRoute::CompleteQr,
                CcmSectorEigenvalueRoute::CrossChecked,
            ] {
                let n = 8;
                let mut f = Fixture::new(n, p, parity, route);
                for i in 0..n {
                    f.matrix[i * n + i] = Float::with_val(p, 5 * i as i32 - 9);
                    for j in (i + 1)..n {
                        let numerator = ((i * 7 + j * 3) % 9) as i32 - 4;
                        let mut entry =
                            Float::with_val(p, if numerator == 0 { 1 } else { numerator });
                        entry /= 32u32;
                        f.matrix[i * n + j] = entry.clone();
                        f.matrix[j * n + i] = entry;
                    }
                }
                let (diagonal, off_diagonal, basis) =
                    xc_numerics::eigen::householder_tridiag_hp_stable(&f.matrix, n, p).unwrap();
                assert!(basis.iter().filter(|x| !x.is_zero()).count() > n * 2);
                f.tri = SectorTridiagonalHp {
                    diagonal,
                    off_diagonal,
                };
                f.transform = SectorTransformHp { basis };
                f.values = compute_sector_eigenvalues(&f.tri, n, f.count, route, p).unwrap();
                let original_allowance = sector_transform_validation::bounds(
                    &f.matrix,
                    &f.tri.diagonal,
                    &f.tri.off_diagonal,
                    &f.transform.basis,
                    n,
                    p,
                )
                .unwrap()
                .eigenvalue_allowance;
                assert!(original_allowance > 0);
                f.allowance =
                    decode_transform_allowance(&encode_transform_allowance(&original_allowance), p)
                        .unwrap();
                assert_eq!(f.allowance.prec(), original_allowance.prec());
                assert_eq!(
                    f.allowance.total_cmp(&original_allowance),
                    std::cmp::Ordering::Equal
                );
                let complete_bounds = sector_resolution::bounds(
                    &f.matrix,
                    &f.tri,
                    &f.transform,
                    &f.values,
                    f.count,
                    p,
                )
                .unwrap();
                f.bounds = sector_resolution::bounds_with_allowance(
                    &f.allowance,
                    &f.tri,
                    &f.values,
                    f.count,
                    p,
                )
                .unwrap();
                for (full, admitted) in complete_bounds.iter().zip(&f.bounds) {
                    for (a, b) in [(&full.0, &admitted.0), (&full.1, &admitted.1)] {
                        assert_eq!(a.prec(), b.prec());
                        assert_eq!(a.total_cmp(b), std::cmp::Ordering::Equal);
                    }
                }
                let old = f.reference().unwrap();
                let new = f.optimized().unwrap();
                assert_eq!(exact_scalars(&old), exact_scalars(&new));
                assert_eq!(
                    serde_json::to_vec(&f.portable(&old)).unwrap(),
                    serde_json::to_vec(&f.portable(&new)).unwrap()
                );
                assert!(f.decode(&f.portable(&new)).is_ok());
                let mut tampered = f.transform.basis.clone();
                tampered[n + 1] += 1;
                assert!(sector_transform_validation::bounds(
                    &f.matrix,
                    &f.tri.diagonal,
                    &f.tri.off_diagonal,
                    &tampered,
                    n,
                    p,
                )
                .is_err());
                assert!(decode_transform_allowance("-1", p).is_err());
                if route != CcmSectorEigenvalueRoute::CompleteQr {
                    f.count = 1;
                    f.bounds.truncate(1);
                    f.values.values[0] += 100;
                    assert_eq!(
                        f.reference().unwrap_err().to_string(),
                        f.optimized().unwrap_err().to_string()
                    );
                }
            }
        }
    }
}
