/* Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.)
 * All rights reserved. See LICENSE in the repository root.
 * Original narrow bridge to separately installed FLINT/Arb, not vendored code.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <gmp.h>
#include <mpfr.h>
#include <flint/flint.h>
#include <flint/arb.h>
#include <flint/arb_hypgeom.h>

#if __FLINT_VERSION != 3
#error "Certified quadrature requires FLINT 3 headers"
#endif
#if !defined(FLINT_USES_TLS) || !FLINT_USES_TLS
#error "Certified quadrature requires FLINT built with thread-local storage"
#endif

/* Arb builds use gmp-mpfr-sys/use-system-libs: the bridge, Rug, and FLINT
 * share the installed GMP/MPFR pair. Refuse header/runtime substitutions.
 * This check supplements, rather than replaces, deployment linkage checks.
 */
#define XC_STRINGIFY_INNER(x) #x
#define XC_STRINGIFY(x) XC_STRINGIFY_INNER(x)
int xc_gl_native_compatible(void)
{
    const char *gmp_headers = XC_STRINGIFY(__GNU_MP_VERSION) "."
        XC_STRINGIFY(__GNU_MP_VERSION_MINOR) "."
        XC_STRINGIFY(__GNU_MP_VERSION_PATCHLEVEL);
    return mpfr_buildopt_tls_p() != 0 &&
        strcmp(gmp_version, gmp_headers) == 0 &&
        strcmp(mpfr_get_version(), MPFR_VERSION_STRING) == 0 &&
        strcmp(flint_version, FLINT_VERSION) == 0;
}

size_t xc_gl_mpfr_size(void) { return sizeof(__mpfr_struct); }
size_t xc_gl_slong_size(void) { return sizeof(slong); }
int64_t xc_gl_emin(void) { return (int64_t)mpfr_get_emin(); }
int64_t xc_gl_emax(void) { return (int64_t)mpfr_get_emax(); }
int xc_gl_threads(void) { return flint_get_num_threads(); }

/* Arb's fast test also certifies MPFR's ternary return value. A ball around
 * an exactly representable value (for example the n=2 weight 1) can have a
 * unique nearest image without determining that ternary value. Outward bounds
 * followed by equal nearest images certify the value alone by monotonicity.
 * The extra endpoint precision is finite and covers the largest guard + 64.
 */
static int xc_gl_unique_nearest(const arb_t value, slong p)
{
    if (!arb_is_finite(value)) return 0;
    if (arb_can_round_mpfr(value, p, MPFR_RNDN)) return 1;
    mpfr_t lower, upper, rounded_lower, rounded_upper;
    mpfr_init2(lower, p+576); mpfr_init2(upper, p+576);
    mpfr_init2(rounded_lower, p); mpfr_init2(rounded_upper, p);
    arb_get_interval_mpfr(lower, upper, value);
    (void)mpfr_set(rounded_lower, lower, MPFR_RNDN);
    (void)mpfr_set(rounded_upper, upper, MPFR_RNDN);
    int accepted = mpfr_number_p(rounded_lower) && mpfr_number_p(rounded_upper) &&
        mpfr_equal_p(rounded_lower, rounded_upper) &&
        (!mpfr_zero_p(rounded_lower) ||
            (arb_is_zero(value) &&
             mpfr_signbit(rounded_lower) == mpfr_signbit(rounded_upper)));
    mpfr_clear(lower); mpfr_clear(upper);
    mpfr_clear(rounded_lower); mpfr_clear(rounded_upper);
    return accepted;
}

/* Status 1: domain/ABI/context; 2: guard exhausted;
 * 4: consistency or rounding failure. Every output is untrusted on failure.
 * The caller owns and initializes all MPFR objects. No global state is changed.
 */
int xc_gl_certified(
    size_t order, uint32_t precision, mpfr_ptr *out_x, mpfr_ptr *out_w)
{
    if (order == 0 || order > 16384 || precision < 64 || precision > 16384 ||
        out_x == NULL || out_w == NULL || flint_get_num_threads() != 1) return 1;
    const slong p = (slong)precision, n = (slong)order;
    for (slong i = 0; i < n; ++i) {
        if (out_x[i] == NULL || out_w[i] == NULL ||
            mpfr_get_prec(out_x[i]) != p || mpfr_get_prec(out_w[i]) != p) return 1;
    }
    arb_ptr nodes = _arb_vec_init(n), weights = _arb_vec_init(n);
    arb_t one, minus_one, two, root, weight, poly, derivative, formula, temp, sum;
    arb_init(one); arb_one(one); arb_init(minus_one); arb_neg(minus_one, one);
    arb_init(two); arb_set_ui(two, 2); arb_init(root); arb_init(weight);
    arb_init(poly); arb_init(derivative); arb_init(formula); arb_init(temp); arb_init(sum);
    arb_zero(sum);
    int status = 0;
    const slong guards[4] = {64, 128, 256, 512};
    for (slong k = 0; k < (n+1)/2; ++k) {
        int accepted = 0;
        for (int rung = 0; rung < 4; ++rung) {
            arb_hypgeom_legendre_p_ui_root(root, weight, (ulong)n, (ulong)k, p+guards[rung]);
            int center = n % 2 && k == n/2;
            if (arb_is_finite(root) && arb_is_finite(weight) && arb_is_positive(weight) &&
                arb_lt(root, one) && arb_gt(root, minus_one) &&
                (center ? arb_is_zero(root) : arb_is_positive(root)) &&
                xc_gl_unique_nearest(root, p) && xc_gl_unique_nearest(weight, p)) {
                if (center) arb_zero(nodes+k);
                else { arb_neg(nodes+k, root); arb_set(nodes+n-1-k, root); }
                arb_set(weights+k, weight); arb_set(weights+n-1-k, weight);
                accepted = 1; break;
            }
        }
        if (!accepted) { status = 2; goto cleanup; }
    }
    for (slong i = 0; i < n; ++i) {
        const slong vp = p+512;
        if (!arb_is_finite(nodes+i) || !arb_is_finite(weights+i) || !arb_is_positive(weights+i) ||
            !arb_gt(nodes+i, minus_one) || !arb_lt(nodes+i, one) ||
            (i && !arb_lt(nodes+i-1, nodes+i)) ||
            !xc_gl_unique_nearest(nodes+i, p) ||
            !xc_gl_unique_nearest(weights+i, p)) { status = 4; goto cleanup; }
        arb_hypgeom_legendre_p_ui(poly, derivative, (ulong)n, nodes+i, vp);
        if (!arb_is_finite(poly) || !arb_contains_zero(poly) || !arb_is_finite(derivative) ||
            arb_contains_zero(derivative)) { status = 4; goto cleanup; }
        arb_mul(temp, nodes+i, nodes+i, vp); arb_sub(temp, one, temp, vp);
        arb_mul(formula, derivative, derivative, vp); arb_mul(formula, formula, temp, vp);
        if (!arb_is_positive(formula)) { status = 4; goto cleanup; }
        arb_div(formula, two, formula, vp);
        if (!arb_is_finite(formula) || !arb_is_positive(formula) ||
            !arb_overlaps(formula, weights+i)) { status = 4; goto cleanup; }
        arb_add(sum, sum, weights+i, vp);
        if (i == n-1-i) {
            if (!arb_is_zero(nodes+i)) { status = 4; goto cleanup; }
            mpfr_set_zero(out_x[i], 1);
        } else (void)arf_get_mpfr(out_x[i], arb_midref(nodes+i), MPFR_RNDN);
        (void)arf_get_mpfr(out_w[i], arb_midref(weights+i), MPFR_RNDN);
        if (!mpfr_number_p(out_x[i]) || !mpfr_number_p(out_w[i]) ||
            mpfr_cmp_ui(out_w[i], 0) <= 0 || mpfr_cmp_si(out_x[i], -1) <= 0 ||
            mpfr_cmp_ui(out_x[i], 1) >= 0 || (i && mpfr_cmp(out_x[i-1],out_x[i]) >= 0)) {
            status = 4; goto cleanup;
        }
    }
    if (!arb_contains(sum, two)) { status = 4; goto cleanup; }
    for (slong i = 0; i < n; ++i) {
        if (!mpfr_equal_p(out_w[i], out_w[n-1-i])) { status = 4; goto cleanup; }
        if (i != n-1-i && mpfr_cmpabs(out_x[i], out_x[n-1-i])) { status = 4; goto cleanup; }
    }
cleanup:
    _arb_vec_clear(nodes,n); _arb_vec_clear(weights,n);
    arb_clear(one); arb_clear(minus_one); arb_clear(two); arb_clear(root); arb_clear(weight);
    arb_clear(poly); arb_clear(derivative); arb_clear(formula); arb_clear(temp); arb_clear(sum);
    return status;
}
