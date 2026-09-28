use serde::Serialize;
use xc_cache::{collect_capture, CaptureFailure, CapturedDiagnostic};

#[derive(Serialize)]
struct Measurement {
    channel: Vec<Option<f64>>,
}

#[test]
fn raw_nonfinite_capture_values_fail_before_collapsing_to_null() {
    for bad in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0xfff0_0000_0000_0001),
    ] {
        assert!(CapturedDiagnostic::new(&bad, vec![]).is_err());
        let nested = Measurement {
            channel: vec![None, Some(bad)],
        };
        assert!(CapturedDiagnostic::new(&nested, vec![]).is_err());
        let record = collect_capture(&17u32, vec!["condition".into()], |_| {
            CapturedDiagnostic::new(&nested, vec![]).map_err(CaptureFailure::failed)
        })
        .unwrap();
        assert!(record.measurements.is_empty());
        let value = serde_json::to_value(record).unwrap();
        assert!(value["receipt"].to_string().contains("failed"));
        assert!(collect_capture(&nested, vec![], |_| unreachable!()).is_err());
    }
    for good in [0.0, -0.0, f64::from_bits(1), f64::MIN_POSITIVE, f64::MAX] {
        let diagnostic = CapturedDiagnostic::new(&Some(good), vec![]).unwrap();
        let restored: Option<f64> = serde_json::from_value(diagnostic.value).unwrap();
        assert_eq!(restored.unwrap().to_bits(), good.to_bits());
    }
    assert_eq!(
        CapturedDiagnostic::new(&Option::<f64>::None, vec![])
            .unwrap()
            .value,
        serde_json::Value::Null
    );
}
