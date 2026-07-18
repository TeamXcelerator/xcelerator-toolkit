/* Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.)
 * All rights reserved. See LICENSE in the repository root.
 *
 * Deliberately narrow C ABI bridge to the system FLINT/Arb shared library.
 * No FLINT source is copied into or statically linked with the toolkit.
 */

#include <mpfr.h>
#include <flint/acb.h>
#include <flint/arb.h>
#include <flint/arb_fmpz_poly.h>
#include <flint/fmpq.h>
#include <flint/fmpq_poly.h>
#include <flint/fmpz_poly.h>

const char *xc_arb_flint_version(void)
{
    return FLINT_VERSION;
}

int xc_flint_rational_polynomial_root_count(
    slong *out_count,
    int *out_square_free,
    mpfr_ptr *out_lowers,
    mpfr_ptr *out_uppers,
    slong output_capacity,
    slong output_precision,
    const char *const *coefficients,
    slong coefficient_count,
    const char *lower,
    const char *upper)
{
    if (out_count == NULL || out_square_free == NULL || coefficients == NULL ||
        coefficient_count < 2 || lower == NULL || upper == NULL ||
        output_precision < 64 ||
        ((out_lowers == NULL) != (out_uppers == NULL)) ||
        (out_lowers != NULL && output_capacity < 1))
        return 1;
    fmpq_poly_t rational_polynomial;
    fmpz_poly_t integer_polynomial;
    fmpq_t value;
    fmpq_t lower_bound;
    fmpq_t upper_bound;
    fmpq_poly_init(rational_polynomial);
    fmpz_poly_init(integer_polynomial);
    fmpq_init(value);
    fmpq_init(lower_bound);
    fmpq_init(upper_bound);
    int status = 0;
    for (slong k = 0; k < coefficient_count; ++k) {
        if (fmpq_set_str(value, coefficients[k], 10) != 0) {
            status = 3;
            goto cleanup;
        }
        fmpq_poly_set_coeff_fmpq(rational_polynomial, k, value);
    }
    if (fmpq_set_str(lower_bound, lower, 10) != 0 ||
        fmpq_set_str(upper_bound, upper, 10) != 0 ||
        fmpq_cmp(lower_bound, upper_bound) >= 0) {
        status = 3;
        goto cleanup;
    }
    fmpq_poly_get_numerator(integer_polynomial, rational_polynomial);
    *out_square_free = fmpz_poly_is_squarefree(integer_polynomial);
    if (!*out_square_free) {
        status = 5;
        goto cleanup;
    }
    const slong degree = fmpz_poly_degree(integer_polynomial);
    acb_ptr roots = _acb_vec_init(degree);
    arb_t lower_ball;
    arb_t upper_ball;
    arb_init(lower_ball);
    arb_init(upper_ball);
    arb_set_fmpq(lower_ball, lower_bound, output_precision);
    arb_set_fmpq(upper_ball, upper_bound, output_precision);
    arb_fmpz_poly_complex_roots(roots, integer_polynomial, 0, output_precision);
    slong count = 0;
    for (slong k = 0; k < degree; ++k) {
        if (acb_is_real(roots + k)) {
            if (arb_gt(acb_realref(roots + k), lower_ball) &&
                arb_lt(acb_realref(roots + k), upper_ball)) {
                if (out_lowers != NULL) {
                    if (count >= output_capacity) {
                        status = 7;
                        break;
                    }
                    arb_get_interval_mpfr(
                        out_lowers[count], out_uppers[count], acb_realref(roots + k));
                }
                ++count;
            } else if (!arb_le(acb_realref(roots + k), lower_ball) &&
                       !arb_ge(acb_realref(roots + k), upper_ball)) {
                status = 2;
                break;
            }
        } else if (arb_contains_zero(acb_imagref(roots + k))) {
            status = 6;
            break;
        }
    }
    *out_count = count;
    arb_clear(upper_ball);
    arb_clear(lower_ball);
    _acb_vec_clear(roots, degree);

cleanup:
    fmpq_clear(upper_bound);
    fmpq_clear(lower_bound);
    fmpq_clear(value);
    fmpz_poly_clear(integer_polynomial);
    fmpq_poly_clear(rational_polynomial);
    return status;
}

void xc_arb_complex_digamma_interval(
    mpfr_ptr out_re_lower,
    mpfr_ptr out_re_upper,
    mpfr_ptr out_im_lower,
    mpfr_ptr out_im_upper,
    mpfr_srcptr in_re_lower,
    mpfr_srcptr in_re_upper,
    mpfr_srcptr in_im_lower,
    mpfr_srcptr in_im_upper,
    slong precision)
{
    acb_t input;
    acb_t output;
    acb_init(input);
    acb_init(output);
    arb_set_interval_mpfr(acb_realref(input), in_re_lower, in_re_upper, precision);
    arb_set_interval_mpfr(acb_imagref(input), in_im_lower, in_im_upper, precision);
    acb_digamma(output, input, precision);
    arb_get_interval_mpfr(out_re_lower, out_re_upper, acb_realref(output));
    arb_get_interval_mpfr(out_im_lower, out_im_upper, acb_imagref(output));
    acb_clear(output);
    acb_clear(input);
}

void xc_arb_complex_trigamma_interval(
    mpfr_ptr out_re_lower,
    mpfr_ptr out_re_upper,
    mpfr_ptr out_im_lower,
    mpfr_ptr out_im_upper,
    mpfr_srcptr in_re_lower,
    mpfr_srcptr in_re_upper,
    mpfr_srcptr in_im_lower,
    mpfr_srcptr in_im_upper,
    slong precision)
{
    acb_t order;
    acb_t input;
    acb_t output;
    acb_init(order);
    acb_init(input);
    acb_init(output);
    acb_one(order);
    arb_set_interval_mpfr(acb_realref(input), in_re_lower, in_re_upper, precision);
    arb_set_interval_mpfr(acb_imagref(input), in_im_lower, in_im_upper, precision);
    acb_polygamma(output, order, input, precision);
    arb_get_interval_mpfr(out_re_lower, out_re_upper, acb_realref(output));
    arb_get_interval_mpfr(out_im_lower, out_im_upper, acb_imagref(output));
    acb_clear(output);
    acb_clear(input);
    acb_clear(order);
}
