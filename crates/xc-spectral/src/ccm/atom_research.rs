//! Authenticated atom tables and finite, explicitly reference-assisted diagnostics.
//! No missing atoms, infinite tails, or displacement identity are inferred.
use super::{
    extended_research::*, research_completion::*, retained_evidence::scalar,
    state_geometry::RetainedState,
};
use anyhow::{bail, Context, Result};
use rug::{Assign, Float};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufRead, BufReader, Read, Seek},
    path::{Component, Path},
};
use xc_cache::ContentDigest;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AtomChunk {
    pub relative_path: String,
    pub sha256: ContentDigest,
    pub bytes: u64,
    pub rows: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct AtomKey {
    pub family: String,
    pub partition: String,
    pub ordinal: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AtomEvaluation {
    pub ordinal: usize,
    pub coordinate: String,
    pub label: String,
    #[serde(default)]
    pub exclude: Vec<AtomKey>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AtomAnalysisPolicy {
    pub maximum_atoms: usize,
    pub maximum_input_bytes: u64,
    #[serde(default)]
    pub weighted_chunks: Vec<AtomChunk>,
    #[serde(default)]
    pub band_chunks: Vec<AtomChunk>,
    #[serde(default)]
    pub evaluations: Vec<AtomEvaluation>,
    #[serde(default)]
    pub cutoffs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tail_recipe: Option<TailFormRecipe>,
}
impl AtomAnalysisPolicy {
    pub fn validate(&self, p: u32) -> Result<()> {
        if self.maximum_atoms == 0
            || self.maximum_input_bytes == 0
            || self.evaluations.len() > 100000
            || self.cutoffs.len() > 32
        {
            bail!("invalid atom analysis limits");
        }
        let mut paths = BTreeSet::new();
        let mut total = 0u64;
        for chunks in [&self.weighted_chunks, &self.band_chunks] {
            let mut rows = 0usize;
            for c in chunks {
                let path = Path::new(&c.relative_path);
                if c.rows == 0
                    || c.bytes == 0
                    || !c.sha256.validate()
                    || path
                        .components()
                        .any(|x| !matches!(x, Component::Normal(_)))
                    || c.relative_path.contains('\\')
                    || !paths.insert(&c.relative_path)
                {
                    bail!("invalid or duplicate atom chunk");
                }
                total = total
                    .checked_add(c.bytes)
                    .context("atom input size overflow")?;
                rows = rows.checked_add(c.rows).context("atom row overflow")?;
            }
            if rows > self.maximum_atoms {
                bail!("atom row limit exceeded");
            }
        }
        if total > self.maximum_input_bytes {
            bail!("atom input byte limit exceeded");
        }
        let mut last = None;
        for c in &self.cutoffs {
            let x = scalar(c, p)?;
            if last.as_ref().is_some_and(|l| &x <= l) {
                bail!("atom cutoffs must be strictly increasing");
            }
            last = Some(x);
        }
        let mut ordinals = BTreeSet::new();
        for e in &self.evaluations {
            scalar(&e.coordinate, p)?;
            if e.ordinal == 0
                || e.label.is_empty()
                || !ordinals.insert(e.ordinal)
                || e.exclude.len() > 1024
            {
                bail!("invalid atom evaluation");
            }
            let mut keys = BTreeSet::new();
            for k in &e.exclude {
                if k.ordinal == 0
                    || k.family.is_empty()
                    || k.partition.is_empty()
                    || !keys.insert(k)
                {
                    bail!("invalid self-atom exclusion");
                }
            }
        }
        Ok(())
    }
}
pub(crate) fn policy(i: Option<&ExternalResearchInputs>) -> Option<&AtomAnalysisPolicy> {
    completion(i).and_then(|c| c.atom_analysis.as_ref())
}

/// Decode one bounded JSON line at a time, after authenticating the entire file.
/// Chunk order is part of the input identity. No executable input is supported.
fn read_chunks<T: serde::de::DeserializeOwned>(
    base: &Path,
    chunks: &[AtomChunk],
    limit: u64,
) -> Result<Vec<T>> {
    let base = base.canonicalize()?;
    let mut result = Vec::new();
    let mut retained = 0u64;
    for c in chunks {
        let path = base.join(&c.relative_path).canonicalize()?;
        if !path.starts_with(&base) {
            bail!("atom chunk escapes input directory");
        }
        let mut f = File::open(path)?;
        if f.metadata()?.len() != c.bytes {
            bail!("atom chunk byte length mismatch");
        }
        let mut hash = Sha256::new();
        let mut buffer = vec![0; 1024 * 1024];
        let mut count = 0u64;
        loop {
            let n = f.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
            count += n as u64;
            if count > c.bytes {
                bail!("atom chunk changed size");
            }
        }
        if count != c.bytes || format!("{:x}", hash.finalize()) != c.sha256.0 {
            bail!("atom chunk SHA-256 mismatch");
        }
        f.rewind()?;
        let mut reader = BufReader::new(f);
        let mut rows = 0;
        let mut line = Vec::new();
        let mut replay = Sha256::new();
        loop {
            line.clear();
            let n = reader
                .by_ref()
                .take(1024 * 1024 + 1)
                .read_until(b'\n', &mut line)?;
            if n == 0 {
                break;
            }
            if n > 1024 * 1024 {
                bail!("atom row exceeds 1 MiB");
            }
            replay.update(&line);
            rows += 1;
            if rows > c.rows {
                bail!("atom chunk row count mismatch");
            }
            retained = retained
                .checked_add(n as u64 + 512)
                .context("atom memory estimate overflow")?;
            if retained > limit {
                bail!("atom input exceeds working memory allowance");
            }
            result.push(serde_json::from_slice(&line)?);
        }
        if rows != c.rows || format!("{:x}", replay.finalize()) != c.sha256.0 {
            bail!("atom chunk changed during decode or has wrong row count");
        }
    }
    Ok(result)
}
pub(crate) fn expand_tables(
    path: &Path,
    atoms: &mut Vec<WeightedAtom>,
    c: Option<&mut CompletionInputs>,
    p: u32,
) -> Result<()> {
    let Some(c) = c else { return Ok(()) };
    let Some(a) = &c.atom_analysis else {
        return Ok(());
    };
    a.validate(p)?;
    let budget =
        super::capture_runtime::CaptureResourcePolicy::from_environment()?.maximum_working_bytes;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    // The input arrays remain resident; half the memory allowance is reserved for
    // decoding/validation and later numerical workspace. The recurrence is bounded separately.
    if !a.weighted_chunks.is_empty() {
        if !atoms.is_empty() {
            bail!("choose inline weighted atoms or chunks, not both");
        }
        *atoms = read_chunks(base, &a.weighted_chunks, budget / 4)?;
    }
    if !a.band_chunks.is_empty() {
        let b = c
            .band
            .as_mut()
            .context("band chunks require a declared band model")?;
        if !b.atoms.is_empty() {
            bail!("choose inline band atoms or chunks, not both");
        }
        b.atoms = read_chunks(base, &a.band_chunks, budget / 4)?;
    }
    Ok(())
}

pub(crate) fn weighted_report(
    s: &RetainedState,
    o: &ExtensionOptions,
    i: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let mut r = weighted_tail_base(s, o, i)?;
    let Some(a) = policy(i) else { return Ok(r) };
    let Some(i) = i.filter(|i| !i.atoms.is_empty()) else {
        return Ok(r);
    };
    let p = o.working_precision_bits;
    let mut groups: BTreeMap<(&str, &str), Vec<usize>> = BTreeMap::new();
    let mut available = BTreeMap::new();
    for (j, x) in i.atoms.iter().enumerate() {
        groups.entry((&x.family, &x.partition)).or_default().push(j);
        if available
            .insert(
                AtomKey {
                    family: x.family.clone(),
                    partition: x.partition.clone(),
                    ordinal: x.ordinal,
                },
                j,
            )
            .is_some()
        {
            bail!("atom keys must be unique within family and partition");
        }
    }
    if r.rows
        .len()
        .saturating_add(a.evaluations.len().saturating_mul(groups.len()))
        > o.maximum_rows
    {
        return Ok(unresolved(r, "atom kernel row budget exceeded"));
    }
    let values = i
        .atoms
        .iter()
        .map(|x| Ok((scalar(&x.coordinate, p)?, scalar(&x.weight, p)?)))
        .collect::<Result<Vec<_>>>()?;
    let tasks = a
        .evaluations
        .iter()
        .flat_map(|e| {
            groups
                .iter()
                .map(move |(group, indices)| (e, group, indices))
        })
        .collect::<Vec<_>>();
    let store = super::capture_runtime::Checkpoints::new(&(
        "signed-atom-kernels-v1",
        &s.manifest.content_digest,
        i,
        o,
    ))?;
    let rows = super::capture_runtime::row_blocks(&store, tasks.len(), |j| {
        let (e, (family, partition), indices) = tasks[j];
        let z = scalar(&e.coordinate, p)?;
        let mut rr = row(e.ordinal, "signed_atom_kernel");
        rr.notes.push(format!("evaluation {}; coordinate {}; family {}; partition {}; supplied ordinal, not inferred zeta identity",e.label,i.atom_coordinate.as_deref().unwrap_or("unspecified"),family,partition));
        put(&mut rr.values, "evaluation_coordinate", &z);
        let exclusions = e
            .exclude
            .iter()
            .map(|k| available.get(k).copied())
            .collect::<Option<BTreeSet<_>>>();
        let Some(exclusions) = exclusions else {
            rr.outcome = "missing_input".into();
            rr.notes
                .push("declared excluded atom absent; no sum emitted".into());
            return Ok(rr);
        };
        let mut sums = [
            Float::with_val(p, 0),
            Float::with_val(p, 0),
            Float::with_val(p, 0),
        ];
        let mut absolute = sums.clone();
        let mut closest: Option<Float> = None;
        let mut excluded = 0;
        let mut used = 0;
        let mut unsafe_denominator = false;
        let guard = (z.clone().abs() + 1u32) >> (p.saturating_sub(32));
        let mut dx = Float::with_val(p, 0);
        let mut distance = dx.clone();
        let mut inverse = dx.clone();
        let mut term = dx.clone();
        let mut magnitude = dx.clone();
        for &index in indices {
            if exclusions.contains(&index) {
                excluded += 1;
                continue;
            }
            let (x, w) = &values[index];
            dx.assign(x);
            dx -= &z;
            distance.assign(&dx);
            distance.abs_mut();
            if closest.as_ref().is_none_or(|v| &distance < v) {
                closest = Some(distance.clone());
            }
            if distance <= guard {
                unsafe_denominator = true;
                continue;
            }
            inverse.assign(1);
            inverse /= &dx;
            term.assign(w);
            for j in 0..3 {
                term *= &inverse;
                sums[j] += &term;
                magnitude.assign(&term);
                magnitude.abs_mut();
                absolute[j] += &magnitude;
            }
            used += 1;
        }
        put(
            &mut rr.values,
            "explicitly_excluded_atoms",
            &Float::with_val(p, excluded),
        );
        put(&mut rr.values, "used_atoms", &Float::with_val(p, used));
        if let Some(v) = closest {
            put(&mut rr.values, "closest_included_atom_distance", &v);
        }
        if unsafe_denominator {
            rr.outcome = "unresolved_denominator".into();
            rr.notes
                .push("coincident or precision-limited included atom: full sums withheld".into());
        } else {
            for j in 0..3 {
                put(
                    &mut rr.values,
                    &format!("signed_kernel_{}", j + 1),
                    &sums[j],
                );
                put(
                    &mut rr.values,
                    &format!("absolute_kernel_{}", j + 1),
                    &absolute[j],
                );
                if sums[j] != 0 {
                    put(
                        &mut rr.values,
                        &format!("cancellation_digits_{}", j + 1),
                        &(Float::with_val(p, &absolute[j]) / sums[j].clone().abs()).log10(),
                    );
                }
            }
        }
        Ok(rr)
    })?;
    if rows.iter().any(|r| r.outcome != "point_measurement") {
        r.outcome = "partial_unresolved".into();
    }
    r.rows.extend(rows);
    r.convention.push_str("; signed kernels sum w/(x-z)^m, m=1..3; explicit exclusions; no displacement identity or infinite-tail assertion");
    Ok(r)
}

fn ladder_row(
    index: usize,
    label: &str,
    cutoff: &str,
    result: Result<ExtendedAnalysis>,
    p: u32,
) -> Result<AnalysisRow> {
    let mut rr = row(index, label);
    put(&mut rr.values, "zero_atom_cutoff", &scalar(cutoff, p)?);
    match result {
        Ok(r) => {
            rr.values.extend(r.values);
            if r.outcome != "point_measurement" {
                rr.outcome = "unresolved_denominator".into();
            }
            if let Some(reason) = r.reason {
                rr.notes.push(reason);
            }
            rr.notes.push(format!("child outcome {}", r.outcome));
            let roots = r
                .rows
                .iter()
                .filter_map(|r| r.values.get("model_band_root"))
                .collect::<Vec<_>>();
            if let (Some(first), Some(last)) = (roots.first(), roots.last()) {
                rr.values
                    .insert("first_model_band_root".into(), (*first).clone());
                rr.values
                    .insert("last_model_band_root".into(), (*last).clone());
            }
            let mut margin: Option<Float> = None;
            for row in r.rows.iter().take(r.rows.len().saturating_sub(1)) {
                if let (Some(n), Some(d)) = (
                    row.values.get("next_signed_norm_squared"),
                    row.values.get("next_absolute_norm_squared"),
                ) {
                    if let (Ok(n), Ok(d)) = (scalar(n, p), scalar(d, p)) {
                        if d > 0 {
                            let v = n / d;
                            if margin.as_ref().is_none_or(|m| &v < m) {
                                margin = Some(v);
                            }
                        }
                    }
                }
            }
            if let Some(v) = margin {
                put(&mut rr.values, "minimum_recurrence_relative_positivity", &v);
            }
        }
        Err(e) => {
            if e.downcast_ref::<std::io::Error>().is_some() {
                return Err(e);
            }
            rr.outcome = "unresolved_denominator".into();
            rr.notes
                .push(format!("finite cutoff diagnostic failed: {e}"));
        }
    }
    Ok(rr)
}
fn append_ladder(r: &mut ExtendedAnalysis, mut rr: AnalysisRow, p: u32) {
    if let Some(previous) = r.rows.last().filter(|x| {
        x.label == rr.label && x.outcome == "point_measurement" && rr.outcome == "point_measurement"
    }) {
        if let Some(c) = previous.values.get("zero_atom_cutoff") {
            rr.values
                .insert("previous_zero_atom_cutoff".into(), c.clone());
        }
        for key in [
            "model_energy",
            "first_model_band_root",
            "last_model_band_root",
            "band_inverse_moment_one",
            "band_inverse_moment_two",
            "band_inverse_moment_three",
        ] {
            if let (Some(a), Some(b)) = (rr.values.get(key), previous.values.get(key)) {
                if let (Ok(a), Ok(b)) = (scalar(a, p), scalar(b, p)) {
                    let delta = a - &b;
                    put(
                        &mut rr.values,
                        &format!("adjacent_signed_change_{key}"),
                        &delta,
                    );
                    if b != 0 {
                        put(
                            &mut rr.values,
                            &format!("adjacent_relative_change_{key}"),
                            &(delta / b.abs()),
                        );
                    }
                }
            }
        }
    }
    if rr.outcome != "point_measurement" {
        r.outcome = "partial_unresolved".into();
    }
    r.rows.push(rr);
}

pub(crate) fn band_report(
    s: &RetainedState,
    o: &ExtensionOptions,
    i: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let mut r = match band_single(s, o, i) {
        Ok(r) => r,
        Err(e) if e.downcast_ref::<std::io::Error>().is_some() => return Err(e),
        Err(e) => unresolved(
            report("band_reconstruction", s, o),
            &format!("band recurrence unavailable: {e}"),
        ),
    };
    let Some(b) = completion(i).and_then(|c| c.band.as_ref()) else {
        return Ok(r);
    };
    let p = o.working_precision_bits;
    put(&mut r.values, "band_degree", &Float::with_val(p, b.degree));
    put(
        &mut r.values,
        "supplied_atom_count",
        &Float::with_val(p, b.atoms.len()),
    );
    let zeros = b
        .atoms
        .iter()
        .filter(|x| x.family == "zero")
        .map(|x| scalar(&x.coordinate, p))
        .collect::<Result<Vec<_>>>()?;
    let last = r
        .rows
        .iter()
        .filter_map(|x| x.values.get("model_band_root"))
        .next_back()
        .map(|v| scalar(v, p))
        .transpose()?;
    if let Some(max) = zeros.iter().max_by(|a, b| a.total_cmp(b)) {
        put(&mut r.values, "largest_supplied_zero_atom", max);
        if let Some(last) = last {
            put(&mut r.values, "largest_model_band_root", &last);
            put(
                &mut r.values,
                "zero_atoms_beyond_model_band",
                &Float::with_val(p, zeros.iter().filter(|x| **x > last).count()),
            );
            put(
                &mut r.values,
                "supplied_zero_extent_covers_model_band",
                &Float::with_val(p, u32::from(max >= &last)),
            );
        }
    }
    let coverage = "zero-atom extent is finite coverage only, not an omitted-tail bound";
    r.reason = Some(match r.reason.take().filter(|s| !s.is_empty()) {
        Some(reason) => format!("{reason}; {coverage}"),
        None => coverage.into(),
    });
    if let Some(a) = policy(i) {
        for (j, c) in a.cutoffs.iter().enumerate() {
            let mut input = i.context("band cutoff input absent")?.clone();
            let model = input
                .run_once
                .as_mut()
                .and_then(|r| r.completion.as_mut())
                .and_then(|c| c.band.as_mut())
                .context("band cutoff model absent")?;
            let cutoff = scalar(c, p)?;
            model.atoms.retain(|x| {
                x.family != "zero" || scalar(&x.coordinate, p).is_ok_and(|v| v <= cutoff)
            });
            model.scoring_roots.clear();
            model.coverage = format!(
                "fixed finite zero cutoff {c}; all supplied nonzero-family atoms retained; {}",
                b.coverage
            );
            let rr = ladder_row(j + 1, "band_cutoff", c, band_single(s, o, Some(&input)), p)?;
            append_ladder(&mut r, rr, p);
        }
    }
    Ok(r)
}
pub(crate) fn tail_report(
    s: &RetainedState,
    o: &ExtensionOptions,
    i: Option<&ExternalResearchInputs>,
) -> Result<ExtendedAnalysis> {
    let a = policy(i);
    let p = o.working_precision_bits;
    let make = |cutoff: Option<&str>| -> Result<super::convergence_capture::RunOnceInputs> {
        let input = i.context("tail recipe input absent")?;
        let recipe = a
            .and_then(|a| a.tail_recipe.as_ref())
            .context("tail recipe absent")?;
        let cut = cutoff.map(|c| scalar(c, p)).transpose()?;
        let atoms = input
            .atoms
            .iter()
            .filter(|x| {
                x.family != "zero"
                    || cut
                        .as_ref()
                        .is_none_or(|c| scalar(&x.coordinate, p).is_ok_and(|x| &x <= c))
            })
            .cloned()
            .collect::<Vec<_>>();
        let form = prepare_tail_form(
            input.definition_digest.clone(),
            &recipe.basis_polynomials,
            &atoms,
            recipe.tail_correction.as_deref(),
            input
                .atom_coverage
                .as_deref()
                .context("tail atom coverage absent")?,
            &recipe.hypotheses,
            p,
        )?;
        Ok(super::convergence_capture::RunOnceInputs {
            tail_form: Some(form),
            ..Default::default()
        })
    };
    let synthesized = if i
        .and_then(|i| i.run_once.as_ref())
        .and_then(|r| r.tail_form.as_ref())
        .is_none()
        && a.is_some_and(|a| a.tail_recipe.is_some())
    {
        Some(make(None)?)
    } else {
        None
    };
    let mut r = super::convergence_capture::tail_operator(
        s,
        o,
        synthesized
            .as_ref()
            .or_else(|| i.and_then(|i| i.run_once.as_ref())),
    )?;
    if let Some(a) = a.filter(|a| a.tail_recipe.is_some()) {
        if synthesized.is_none() {
            r.reason=Some("primary uses the explicitly supplied tail form; cutoff rows use the separately declared recipe. No primary-to-ladder basis equality is inferred".into());
        }
        for (j, c) in a.cutoffs.iter().enumerate() {
            let child = make(Some(c))
                .and_then(|once| super::convergence_capture::tail_operator(s, o, Some(&once)));
            let mut rr = ladder_row(j + 1, "tail_model_cutoff", c, child, p)?;
            rr.notes.push("same supplied basis, lattice and tail correction across cutoffs; finite cutoff stability is not an infinite-tail bound".into());
            append_ladder(&mut r, rr, p);
        }
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn temp() -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "xc-atom-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&p).unwrap();
        p
    }
    #[test]
    fn cutoff_row_rejects_invalid_scalar_without_panicking() {
        assert!(ladder_row(
            1,
            "band_cutoff",
            "not-a-number",
            Err(anyhow::anyhow!("unused")),
            128
        )
        .is_err());
    }
    #[test]
    fn chunks_verify_bytes_rows_paths_and_allow_more_than_old_atom_limit() {
        let root = temp();
        let line = b"{\"coordinate\":\"1\",\"signed_weight\":\"1\",\"family\":\"zero\"}\n";
        let bytes = line.repeat(200001);
        std::fs::write(root.join("part.jsonl"), &bytes).unwrap();
        let chunk = AtomChunk {
            relative_path: "part.jsonl".into(),
            sha256: ContentDigest::sha256(&bytes),
            bytes: bytes.len() as u64,
            rows: 200001,
        };
        let policy = AtomAnalysisPolicy {
            maximum_atoms: 250000,
            maximum_input_bytes: 64 << 20,
            weighted_chunks: vec![],
            band_chunks: vec![chunk.clone()],
            evaluations: vec![],
            cutoffs: vec![],
            tail_recipe: None,
        };
        policy.validate(128).unwrap();
        let got = read_chunks::<BandAtom>(&root, std::slice::from_ref(&chunk), 256 << 20).unwrap();
        assert_eq!(got.len(), 200001);
        assert!(read_chunks::<BandAtom>(&root, std::slice::from_ref(&chunk), 100).is_err());
        let mut bad = chunk.clone();
        bad.rows -= 1;
        assert!(read_chunks::<BandAtom>(&root, &[bad], 256 << 20).is_err());
        let mut bad = policy;
        bad.band_chunks[0].relative_path = "../part.jsonl".into();
        assert!(bad.validate(128).is_err());
        std::fs::write(root.join("part.jsonl"), b"broken").unwrap();
        assert!(read_chunks::<BandAtom>(&root, &[chunk], 256 << 20).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
