use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use xc_cache::ContentDigest;
use xc_core::DecimalLiteral;
use xc_root::IntervalNewtonOptions;
use xc_spectral::ccm::{
    certified_roots::{
        certify_production_first_positive_ccm_roots,
        compare_production_first_positive_roots_to_references,
        verify_production_first_positive_ccm_root_certificate,
        verify_production_post_discovery_reference_comparison, PostDiscoveryReferenceComparison,
        ProductionFirstPositiveCcmRootCertificate, ReferenceZeroDataset,
    },
    hp::{self, HighPrecConfig, PortableHighPrecResult},
    CcmParams,
};
use xc_zeta::zeros::{BUNDLED_ZETA_ZEROS_JSON, BUNDLED_ZETA_ZEROS_RESOURCE};

const CUTOFF: u64 = 13;
const MODES: usize = 120;
const DECIMAL_DIGITS: u32 = 200;
const ROOTS: usize = 50;
const REFERENCE_SHA256: &str = "10c022e1c912b67d3ae281c7ad069811ed39b84478c13e167f9d00a067a87224";
const REFERENCE_CITATION: &str = "Computed to 2,500 significant digits using rigorous Arb interval arithmetic (F. Johansson, Arb: efficient arbitrary-precision midpoint-radius interval arithmetic, IEEE Trans. Comput. 66 (2017), 1281-1292); leading 1,000 digits independently cross-checked against A. M. Odlyzko's standard zeta-zero tabulation";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProductionFirstFiftyEvidence {
    schema_version: u32,
    toolkit_version: String,
    source_method: String,
    source_cache_mode: String,
    source_reference_seed_count: usize,
    source_result: PortableHighPrecResult,
    certificate: ProductionFirstPositiveCcmRootCertificate,
    reference_resource: String,
    reference_file_sha256: ContentDigest,
    reference_dataset: ReferenceZeroDataset,
    comparison: PostDiscoveryReferenceComparison,
}

fn interval_options() -> Result<IntervalNewtonOptions> {
    Ok(IntervalNewtonOptions {
        width_tolerance: DecimalLiteral::new("1e-50")?,
        maximum_iterations: 50,
    })
}

fn reference_dataset(bytes: &[u8]) -> Result<ReferenceZeroDataset> {
    let ordinates: Vec<String> = serde_json::from_slice(bytes).context("parse reference zeros")?;
    if ordinates.len() < ROOTS {
        bail!("reference file contains fewer than {ROOTS} positive ordinates");
    }
    ReferenceZeroDataset::new(
        "arb-certified-zeta-zeros-first-50",
        REFERENCE_CITATION,
        format!("bundled_resource={BUNDLED_ZETA_ZEROS_RESOURCE};sha256={REFERENCE_SHA256}"),
        8_320,
        ordinates.into_iter().take(ROOTS).collect(),
    )
}

fn replay_certificate(
    source_result: &PortableHighPrecResult,
) -> Result<ProductionFirstPositiveCcmRootCertificate> {
    let runtime = source_result.to_runtime()?;
    if !runtime.eigenvalues_pos.is_empty()
        || runtime.xi.len() != 2 * MODES + 1
        || runtime.precision_bits != source_result.precision_bits
    {
        bail!("stored production source has invalid spectrum, dimension, or precision");
    }
    certify_production_first_positive_ccm_roots(
        &runtime.xi,
        CUTOFF,
        MODES,
        ROOTS,
        runtime.precision_bits,
        96,
        &interval_options()?,
    )
}

fn verify(evidence: &ProductionFirstFiftyEvidence) -> Result<()> {
    let bundled_digest = ContentDigest::sha256(BUNDLED_ZETA_ZEROS_JSON);
    if evidence.schema_version != 1
        || evidence.toolkit_version != env!("CARGO_PKG_VERSION")
        || evidence.source_method != "xc_spectral::ccm::hp::build_source"
        || evidence.source_cache_mode != "off_pure_compute"
        || evidence.source_reference_seed_count != 0
        || evidence.reference_resource != BUNDLED_ZETA_ZEROS_RESOURCE
        || bundled_digest.to_string() != REFERENCE_SHA256
        || evidence.reference_file_sha256 != bundled_digest
    {
        bail!("production first-50 evidence identity or provenance is invalid");
    }
    let replay = replay_certificate(&evidence.source_result)?;
    if replay != evidence.certificate {
        bail!("production certificate does not replay from the stored computed Weil source");
    }
    verify_production_first_positive_ccm_root_certificate(&evidence.certificate)?;
    let references = reference_dataset(BUNDLED_ZETA_ZEROS_JSON)?;
    if references != evidence.reference_dataset {
        bail!("reference dataset does not replay from the provenance-bound source file");
    }
    verify_production_post_discovery_reference_comparison(
        &evidence.comparison,
        &evidence.certificate,
        &evidence.reference_dataset,
    )?;
    Ok(())
}

fn generate(output_path: &Path) -> Result<()> {
    let params = CcmParams::from_lambda_sq_integer(CUTOFF, MODES);
    let mut config = HighPrecConfig::for_decimal_digits(DECIMAL_DIGITS);
    config.cache_mode = xc_numerics::quadrature::CacheMode::Off;
    let result = hp::build_source(&params, &config)?;
    let source_result = PortableHighPrecResult::from_runtime(&result)?;
    let certificate = replay_certificate(&source_result)?;
    let references = reference_dataset(BUNDLED_ZETA_ZEROS_JSON)?;
    let comparison =
        compare_production_first_positive_roots_to_references(&certificate, &references)?;
    let evidence = ProductionFirstFiftyEvidence {
        schema_version: 1,
        toolkit_version: env!("CARGO_PKG_VERSION").to_owned(),
        source_method: "xc_spectral::ccm::hp::build_source".to_owned(),
        source_cache_mode: "off_pure_compute".to_owned(),
        source_reference_seed_count: 0,
        source_result,
        certificate,
        reference_resource: BUNDLED_ZETA_ZEROS_RESOURCE.to_owned(),
        reference_file_sha256: ContentDigest::sha256(BUNDLED_ZETA_ZEROS_JSON),
        reference_dataset: references,
        comparison,
    };
    std::fs::write(output_path, serde_json::to_vec_pretty(&evidence)?)
        .context("write production first-50 evidence")?;
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let command = args
        .next()
        .context("usage: ccm_first_fifty_production <generate|verify> <evidence.json>")?;
    let evidence_path = PathBuf::from(args.next().context("missing evidence path")?);
    if args.next().is_some() {
        bail!("unexpected extra argument");
    }
    match command.to_string_lossy().as_ref() {
        "generate" => generate(&evidence_path),
        "verify" => {
            let evidence: ProductionFirstFiftyEvidence =
                serde_json::from_slice(&std::fs::read(evidence_path)?)?;
            verify(&evidence)
        }
        other => bail!("unknown command {other:?}; expected generate or verify"),
    }
}
