use xc_spectral::ccm::window::{plan_observation, ZeroTarget};

fn main() {
    let plan = plan_observation(100.0, &ZeroTarget::FirstK { count: 100 }, 1000, 150).unwrap();
    println!("{}", serde_json::to_string_pretty(&plan).unwrap());
}
