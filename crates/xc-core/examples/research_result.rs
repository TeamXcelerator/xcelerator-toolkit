use xc_core::{Diagnostics, EvidenceRef, ResearchResult, SolverProvenance};

fn main() {
    let mut result = ResearchResult::computed(
        "9.564814428856e-464".to_owned(),
        SolverProvenance::current_package("rug_mpfr"),
    );
    result.diagnostics = Diagnostics::default();
    result
        .diagnostics
        .insert_scalar("scaled_backward_error", "4.5e-1005");
    result.add_evidence(EvidenceRef::new(
        "higher_precision_repeat",
        "repeat-8192-bit.json",
        "higher-precision stability repeat",
    ));
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}
