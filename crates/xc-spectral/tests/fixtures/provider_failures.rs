// A standalone protocol fixture compiled by target_provider_failures.rs.
// The mathematical function is f(u) = u; one scalar reply is deliberately bad.
use std::io::{self, BufRead, Write};

fn field<'a>(line: &'a str, name: &str) -> &'a str {
    let marker = format!("\"{name}\":");
    line.split_once(&marker)
        .unwrap()
        .1
        .split([',', '}'])
        .next()
        .unwrap()
}

fn main() {
    let counter = std::env::var_os("XC_TEST_PROVIDER_COUNTER").unwrap();
    let bad = std::env::var("XC_TEST_PROVIDER_BAD_SCALAR").unwrap();
    let mut requests = 0;
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let id = field(&line, "request_id");
        let nonce = field(&line, "request_nonce");
        let operation = field(&line, "operation");
        let bits = field(&line, "precision_bits");
        let mut value = "\"1\"".to_owned();
        if operation == "\"evaluate\"" {
            value = field(&line, "u").to_owned();
            if value.trim_matches('"').parse::<f64>().unwrap() == 2.0 {
                requests += 1;
                std::fs::write(&counter, requests.to_string()).unwrap();
                if requests == 1 {
                    value = format!("\"{bad}\"");
                }
            }
        }
        println!(
            "{{\"protocol_version\":2,\"request_id\":{id},\"request_nonce\":{nonce},\"operation\":{operation},\"precision_bits\":{bits},\"ready\":true,\"value\":{value}}}"
        );
        io::stdout().flush().unwrap();
    }
}
