use xc_core::{AssuranceLevel, EigenTarget, PrecisionPolicy};
use xc_operator::MatrixStructure;
use xc_solver::{plan_symmetric_eigenproblem, SolverPlannerInput};

fn main() {
    let request = SolverPlannerInput {
        structure: MatrixStructure::Dense,
        dimension: 401,
        target: EigenTarget::IndexRange { first: 0, last: 2 },
        requested_eigenpairs: 3,
        assurance: AssuranceLevel::CrossChecked,
        precision: PrecisionPolicy {
            initial_bits: 4096,
            maximum_bits: 8192,
            guard_bits: 128,
            escalation: xc_core::PrecisionEscalation::Multiply {
                numerator: 2,
                denominator: 1,
            },
        },
        matrix_materialized: true,
        generalized: false,
    };
    let plan = plan_symmetric_eigenproblem(&request).unwrap();
    println!("{}", serde_json::to_string_pretty(&plan).unwrap());
}
