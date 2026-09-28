#![cfg(feature = "hp")]
use rug::Float;
use xc_spectral::screw::ScrewKernel;
fn hp(s: &str) -> Float {
    Float::with_val(256, Float::parse(s).unwrap())
}
#[test]
fn screw_matches_published_sqrt_prime_formula() {
    let kernel = ScrewKernel::new(2.0, 256);
    let cases = [
        ("0.1", "-0.053130433817770254100052344091465441847551560737687439841439796039723450676786886204400194444682"),
        ("0.5", "-0.04016258038208666498611414799699111911057509397533187891651305278774977827185739694551057801245"),
        ("0.7", "-0.061766283929482370616195657089678040917781683883321116721498539197259328198279772421523632147433"),
        ("1", "-0.044007305236852526860218606408083772922687241744930873163422724816625908153756073805344700682245"),
        ("1.3", "-0.038051765345468303856907782341990310938341428574467447142839022070640833355433078883929453694652"),
        ("2.5", "-0.048434868673898385098113853248465872741517281005273632964992079581229773638501528956611409456007"),
        ("4", "-0.03514093054519634002609317798932124403683291759285596000611987887544740861634821405893983865409"),
    ];
    for (t, expected) in cases {
        let difference = (kernel.eval(&hp(t)) - hp(expected)).abs();
        assert!(
            difference < hp("1e-65"),
            "published g({t}) mismatch: {difference}"
        );
    }
}

#[test]
fn screw_cusp_and_support_are_checked() {
    let kernel = ScrewKernel::new(2.0, 256);
    let cases = [
        ("1e-20","-0.00000000000000000022318304564285017668105078765396823220628538091902206770335466097410248759679654291418673623935"),
        ("1e-50","-5.6857080959195702928366200585662286334645060074111587938613163941467924615323584761531979469856e-49"),
    ];
    for (t, expected) in cases {
        let error = (kernel.try_eval(&hp(t)).unwrap() - hp(expected)).abs();
        assert!(error < hp(t) * hp("1e-58"), "cusp mismatch at {t}: {error}");
    }
    assert!(kernel.try_eval(&hp("4.0001")).is_err());
    assert!(kernel.eval(&hp("4.0001")).is_nan());
    assert!(kernel
        .try_eval(&Float::with_val(256, rug::float::Special::Nan))
        .is_err());
    assert!(ScrewKernel::try_new(f64::NAN, 256).is_err());
    assert!(ScrewKernel::try_new(-1.0, 256).is_err());
    assert!(ScrewKernel::try_new(1000.0, 256).is_err());
    assert!(ScrewKernel::try_new(1.0, 0).is_err());
}

#[test]
fn exhaustive_resumed_screw_checked_constructor_does_not_panic_on_capacity() {
    // exp(44) fits u64/usize but exceeds Vec's isize byte capacity on 64-bit platforms.
    let result = std::panic::catch_unwind(|| ScrewKernel::try_new(22.0, 64));
    assert!(result.is_ok(), "fallible screw constructor panicked");
    assert!(result.unwrap().is_err());
}
