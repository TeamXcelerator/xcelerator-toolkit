/* Copyright (c) 2026 Ronnie Andrews, Jr. (Team Xcelerator Inc.)
 * All rights reserved. See LICENSE in the repository root.
 *
 * Deliberately narrow C ABI bridge to the system FLINT/Arb shared library.
 * No FLINT source is copied into or statically linked with the toolkit.
 */

#include <stdint.h>
#include <stddef.h>
#include <mpfr.h>
#include <flint/acb.h>
#include <flint/acb_hypgeom.h>
#include <flint/acb_poly.h>
#include <flint/arb.h>
#include <flint/arb_fmpz_poly.h>
#include <flint/fmpq.h>
#include <flint/fmpq_poly.h>
#include <flint/fmpz_poly.h>

/* Rust uses isize for every slong argument and output slot. */
typedef char xc_flint_slong_pointer_width[
    (sizeof(slong) == sizeof(intptr_t)) ? 1 : -1];

size_t xc_arb_slong_size(void)
{
    return sizeof(slong);
}

const char *xc_arb_flint_version(void)
{
    return FLINT_VERSION;
}

/* Exact secular numerator only. A node stores integer polynomials A,B with
 * D=A/2^d and N=B/2^(d+e), where e is the common residue denominator exponent.
 * A balanced merge is (A1*A2, B1*A2+A1*B2). Native integer multiplication
 * avoids a rational normalization for every scalar product and every pole. */
static int xc_exact_secular_tree(fmpz_poly_t denominator, fmpz_poly_t numerator,
    slong *denominator_bits, mpfr_srcptr const *poles,
    mpfr_srcptr const *weights, slong count, slong weight_bits, slong total_bits,
    int need_denominator)
{
    if (count == 1) {
        mpq_t pole, weight;
        fmpz_t coefficient;
        mpq_init(pole); mpq_init(weight); fmpz_init(coefficient);
        mpfr_get_q(pole, poles[0]);
        mpfr_get_q(weight, weights[0]);
        slong pole_bits = (slong)mpz_sizeinbase(mpq_denref(pole), 2) - 1;
        slong residue_bits = (slong)mpz_sizeinbase(mpq_denref(weight), 2) - 1;
        int status = 0;
        if (pole_bits > total_bits || residue_bits > weight_bits ||
            pole_bits > total_bits - weight_bits) {
            status = 3;
        } else {
            if (need_denominator) {
                fmpz_set_mpz(coefficient, mpq_numref(pole));
                fmpz_neg(coefficient, coefficient);
                fmpz_poly_set_coeff_fmpz(denominator, 0, coefficient);
                fmpz_set_mpz(coefficient, mpq_denref(pole));
                fmpz_poly_set_coeff_fmpz(denominator, 1, coefficient);
            }
            fmpz_set_mpz(coefficient, mpq_numref(weight));
            fmpz_mul_2exp(coefficient, coefficient,
                (ulong)(pole_bits + weight_bits - residue_bits));
            fmpz_poly_set_coeff_fmpz(numerator, 0, coefficient);
            *denominator_bits = pole_bits;
        }
        fmpz_clear(coefficient); mpq_clear(weight); mpq_clear(pole);
        return status;
    }
    fmpz_poly_t left_d, left_n, right_d, right_n, other;
    fmpz_poly_init(left_d); fmpz_poly_init(left_n);
    fmpz_poly_init(right_d); fmpz_poly_init(right_n); fmpz_poly_init(other);
    slong left_bits = 0, right_bits = 0;
    slong middle = count / 2;
    int status = xc_exact_secular_tree(left_d, left_n, &left_bits,
        poles, weights, middle, weight_bits, total_bits, 1);
    if (status == 0)
        status = xc_exact_secular_tree(right_d, right_n, &right_bits,
            poles + middle, weights + middle, count - middle, weight_bits, total_bits, 1);
    if (status == 0 && left_bits > total_bits - weight_bits - right_bits)
        status = 3;
    if (status == 0) {
        if (need_denominator)
            fmpz_poly_mul(denominator, left_d, right_d);
        fmpz_poly_mul(numerator, left_n, right_d);
        fmpz_poly_mul(other, left_d, right_n);
        fmpz_poly_add(numerator, numerator, other);
        *denominator_bits = left_bits + right_bits;
    }
    fmpz_poly_clear(other); fmpz_poly_clear(right_n); fmpz_poly_clear(right_d);
    fmpz_poly_clear(left_n); fmpz_poly_clear(left_d);
    return status;
}

int xc_flint_exact_secular_numerator(mpq_ptr *out_coefficients, slong output_capacity,
    mpfr_srcptr const *poles, mpfr_srcptr const *weights, slong count,
    slong weight_bits, slong total_bits)
{
    if (out_coefficients == NULL || poles == NULL || weights == NULL ||
        count < 1 || output_capacity != count || weight_bits < 0 || total_bits < weight_bits)
        return 1;
    for (slong k = 0; k < count; ++k) {
        if (out_coefficients[k] == NULL || poles[k] == NULL || weights[k] == NULL ||
            !mpfr_number_p(poles[k]) || !mpfr_number_p(weights[k]) || mpfr_zero_p(weights[k]))
            return 1;
    }
    fmpz_poly_t denominator, numerator;
    fmpz_t coefficient;
    fmpz_poly_init(denominator); fmpz_poly_init(numerator); fmpz_init(coefficient);
    slong denominator_bits = 0;
    int status = xc_exact_secular_tree(denominator, numerator, &denominator_bits,
        poles, weights, count, weight_bits, total_bits, 0);
    if (status == 0 && denominator_bits != total_bits - weight_bits)
        status = 3;
    if (status == 0) {
        for (slong k = 0; k < count; ++k) {
            fmpz_poly_get_coeff_fmpz(coefficient, numerator, k);
            if (fmpz_is_zero(coefficient)) {
                mpq_set_ui(out_coefficients[k], 0, 1);
                continue;
            }
            fmpz_get_mpz(mpq_numref(out_coefficients[k]), coefficient);
            /* The denominator is a power of two. Remove its common factors
             * directly, producing canonical GMP rationals without a GCD or
             * a large denominator allocation for an exactly zero coefficient. */
            mp_bitcnt_t twos = mpz_scan1(mpq_numref(out_coefficients[k]), 0);
            if (twos > (ulong)total_bits) twos = (ulong)total_bits;
            mpz_tdiv_q_2exp(mpq_numref(out_coefficients[k]),
                mpq_numref(out_coefficients[k]), twos);
            mpz_set_ui(mpq_denref(out_coefficients[k]), 1);
            mpz_mul_2exp(mpq_denref(out_coefficients[k]),
                mpq_denref(out_coefficients[k]), (ulong)total_bits - twos);
        }
    }
    fmpz_clear(coefficient); fmpz_poly_clear(numerator); fmpz_poly_clear(denominator);
    return status;
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

/* Entire-series coefficients of sinc(q + h*t), used only when |q| <= 1.
 * From sinc(q) = (1/2) integral_{-1}^1 exp(i*q*x) dx, coefficient k is
 * (-1)^(k/2) h^k/k! sum_j (-q^2)^j/((2j)! (k+2j+1)) for even k;
 * for odd k it is (-1)^((k+1)/2) q h^k/k! times the analogous sum with
 * (2j+1)! (k+2j+2). After J terms either inner tail is <= 2/(2J)!:
 * |q| <= 1 and consecutive factorial terms have ratio at most 1/2.
 * Add that error BEFORE the odd q factor, preserving the removable limit
 * and relative accuracy even for tiny nonzero q. All operations are balls. */
static void sinc_near_carrier(acb_ptr coefficients, const acb_t q,
    const arb_t h, slong order, slong precision, arb_srcptr powers)
{
    acb_t q2, even, odd, term;
    arb_t factorial, error, tolerance, power;
    acb_init(q2); acb_init(even); acb_init(odd); acb_init(term);
    arb_init(factorial); arb_init(error); arb_init(tolerance); arb_init(power);
    _acb_vec_zero(coefficients, order);
    acb_mul(q2, q, q, precision); acb_neg(q2, q2);
    acb_one(even); arb_one(factorial);
    arb_one(tolerance); arb_mul_2exp_si(tolerance, tolerance, -precision - 8);
    for (slong j = 0; ; j++) {
        acb_div_ui(odd, even, 2*j + 1, precision);
        for (slong k = 0; k < order; k++) {
            acb_div_ui(term, k % 2 ? odd : even, k + 2*j + 1 + k%2, precision);
            acb_add(coefficients + k, coefficients + k, term, precision);
        }
        acb_mul(even, even, q2, precision);
        acb_div_ui(even, even, 2*j + 1, precision);
        acb_div_ui(even, even, 2*j + 2, precision);
        arb_mul_ui(factorial, factorial, 2*j + 1, precision);
        arb_mul_ui(factorial, factorial, 2*j + 2, precision);
        arb_ui_div(error, 2, factorial, precision);
        if (arb_lt(error, tolerance)) break;
    }
    arb_one(power);
    for (slong k = 0; k < order; k++) {
        acb_add_error_arb(coefficients + k, error);
        if (k % 2) acb_mul(coefficients + k, coefficients + k, q, precision);
        if ((k + 1) % 4 >= 2) acb_neg(coefficients + k, coefficients + k);
        acb_mul_arb(coefficients + k, coefficients + k,
            powers ? powers + k : power, precision);
        if (!powers) {
            arb_mul(power, power, h, precision);
            arb_div_ui(power, power, k + 1, precision);
        }
    }
    acb_clear(q2); acb_clear(even); acb_clear(odd); acb_clear(term);
    arb_clear(factorial); arb_clear(error); arb_clear(tolerance); arb_clear(power);
}

static int near_carrier(const acb_t q, slong precision)
{
    arb_t magnitude, one;
    arb_init(magnitude); arb_init(one); arb_one(one);
    acb_abs(magnitude, q, precision);
    int near = arb_le(magnitude, one);
    arb_clear(magnitude); arb_clear(one);
    return near;
}

/* This object owns only immutable balls, never pointers into the Rust source.
 * Every evaluation owns its mutable temporaries. Prepared entries are exactly
 * the balls produced at the corresponding sites in the one-shot operation
 * sequence; neither multiplication order nor accumulation order is changed. */
typedef struct {
    slong count, precision;
    arb_t l, half, pi, root_l, factorial48, factorial47;
    arb_ptr coefficients, slope_coefficients, offsets, powers;
} xc_arb_finite_transform_context;

void xc_arb_finite_transform_context_free(void *opaque)
{
    if (!opaque) return;
    xc_arb_finite_transform_context *context = opaque;
    _arb_vec_clear(context->coefficients, context->count);
    _arb_vec_clear(context->slope_coefficients, context->count);
    _arb_vec_clear(context->offsets, context->count);
    _arb_vec_clear(context->powers, 48);
    arb_clear(context->l); arb_clear(context->half); arb_clear(context->pi);
    arb_clear(context->root_l); arb_clear(context->factorial48);
    arb_clear(context->factorial47);
    flint_free(context);
}

void *xc_arb_finite_transform_context_new(const char *cutoff,
    mpfr_srcptr const *coefficients, slong count, slong precision)
{
    arb_t l, half, pi, norm, c, scale;
    arb_init(l); arb_init(half); arb_init(pi); arb_init(norm);
    arb_init(c); arb_init(scale);
    xc_arb_finite_transform_context *context = NULL;
    if (count <= 0 || count % 2 == 0 || arb_set_str(l, cutoff, precision) != 0
        || !arb_is_finite(l)) goto cleanup;
    arb_sub_ui(c, l, 1, precision); if (!arb_is_positive(c)) goto cleanup;
    arb_log(l, l, precision); arb_mul_2exp_si(half, l, -1);
    arb_const_pi(pi, precision); arb_zero(norm);
    for (slong j = 0; j < count; j++) {
        arb_set_interval_mpfr(c, coefficients[j], coefficients[j], precision);
        arb_addmul(norm, c, c, precision);
    }
    if (!arb_is_finite(norm) || !arb_is_positive(norm)) goto cleanup;
    arb_div(scale, l, norm, precision); arb_sqrt(scale, scale, precision);
    context = flint_malloc(sizeof(xc_arb_finite_transform_context));
    context->count = count; context->precision = precision;
    arb_init(context->l); arb_init(context->half); arb_init(context->pi);
    arb_init(context->root_l); arb_init(context->factorial48);
    arb_init(context->factorial47);
    arb_set(context->l, l); arb_set(context->half, half); arb_set(context->pi, pi);
    arb_sqrt(context->root_l, l, precision);
    arb_fac_ui(context->factorial48, 48, precision);
    arb_fac_ui(context->factorial47, 47, precision);
    context->coefficients = _arb_vec_init(count);
    context->slope_coefficients = _arb_vec_init(count);
    context->offsets = _arb_vec_init(count);
    context->powers = _arb_vec_init(48);
    for (slong j = 0; j < count; j++) {
        slong mode = j - count / 2;
        arb_mul_si(context->offsets + j, pi, mode, precision);
        arb_set_interval_mpfr(c, coefficients[j], coefficients[j], precision);
        arb_mul(c, c, scale, precision);
        if (mode % 2) arb_neg(c, c);
        arb_set(context->coefficients + j, c);
        arb_mul(c, c, half, precision);
        arb_set(context->slope_coefficients + j, c);
    }
    arb_one(c);
    for (slong k = 0; k < 48; k++) {
        arb_set(context->powers + k, c);
        arb_mul(c, c, half, precision);
        arb_div_ui(c, c, k + 1, precision);
    }
cleanup:
    arb_clear(l); arb_clear(half); arb_clear(pi); arb_clear(norm);
    arb_clear(c); arb_clear(scale);
    return context;
}

/* Centered Taylor enclosure of a finite unit coefficient transform. Summing
 * Taylor coefficients before evaluating the rectangle preserves cancellation
 * between Fourier carriers. The integral remainder uses ||f||_1 <= sqrt(L),
 * |x| <= L/2, and |exp(i*z*x)| <= exp(|Im(z)|*L/2). It bounds both value
 * and derivative, including all omitted Taylor terms, with Arb arithmetic. */
static int finite_transform_taylor(acb_t value, acb_t slope, const acb_t z,
    const arb_t l, const arb_t half, const arb_t pi, const arb_t scale,
    mpfr_srcptr const *coefficients, slong count, slong precision,
    const xc_arb_finite_transform_context *context)
{
    const slong order = 48;
    acb_poly_t q, carrier;
    acb_ptr sum = _acb_vec_init(order);
    acb_ptr near_coefficients = _acb_vec_init(order);
    acb_t mid, delta, term, q0, sine, cosine, previous, coefficient;
    arb_t c, radius, extent, error, factorial, envelope, power;
    acb_poly_init(q); acb_poly_init(carrier);
    acb_init(mid); acb_init(delta); acb_init(term); acb_init(q0);
    acb_init(sine); acb_init(cosine); acb_init(previous); acb_init(coefficient);
    arb_init(c); arb_init(radius); arb_init(extent); arb_init(error);
    arb_init(factorial); arb_init(envelope); arb_init(power);
    acb_get_mid(mid, z);
    acb_sub(delta, z, mid, precision);
    for (slong j = 0; j < count; j++) {
        slong mode = j - count / 2;
        acb_mul_arb(q0, mid, half, precision);
        if (context) arb_set(c, context->offsets + j);
        else arb_mul_si(c, pi, mode, precision);
        arb_add(acb_realref(q0), acb_realref(q0), c, precision);
        int near = near_carrier(q0, precision);
        int removable = acb_contains_zero(q0);
        if (near) {
            sinc_near_carrier(near_coefficients, q0, half, order, precision,
                context ? context->powers : NULL);
        } else if (removable) {
            acb_poly_set_coeff_acb(q, 0, q0);
            acb_set_arb(term, half);
            acb_poly_set_coeff_acb(q, 1, term);
            acb_poly_sinc_series(carrier, q, order, precision);
        } else {
            /* Explicit scalar recurrence avoids precision loss in the installed
             * polynomial transcendental-series implementation. */
            acb_sin_cos(sine, cosine, q0, precision);
        }
        if (context) arb_set(c, context->coefficients + j);
        else {
            arb_set_interval_mpfr(c, coefficients[j], coefficients[j], precision);
            arb_mul(c, c, scale, precision);
            if (mode % 2) arb_neg(c, c);
        }
        arb_one(power);
        acb_zero(previous);
        for (slong k = 0; k < order; k++) {
            if (near) {
                acb_set(coefficient, near_coefficients + k);
            } else if (removable) {
                acb_poly_get_coeff_acb(coefficient, carrier, k);
            } else {
                acb_set(term, k % 2 ? cosine : sine);
                if (k % 4 >= 2) acb_neg(term, term);
                acb_mul_arb(term, term,
                    context ? context->powers + k : power, precision);
                acb_mul_arb(coefficient, previous, half, precision);
                acb_sub(coefficient, term, coefficient, precision);
                acb_div(coefficient, coefficient, q0, precision);
                acb_set(previous, coefficient);
            }
            acb_mul_arb(term, coefficient, c, precision);
            acb_add(sum + k, sum + k, term, precision);
            if (!context) {
                arb_mul(power, power, half, precision);
                arb_div_ui(power, power, k + 1, precision);
            }
        }
    }
    acb_set(value, sum + order - 1);
    acb_zero(slope);
    for (slong k = order - 2; k >= 0; k--) {
        acb_mul(value, value, delta, precision);
        acb_add(value, value, sum + k, precision);
        acb_mul(slope, slope, delta, precision);
        acb_mul_ui(term, sum + k + 1, k + 1, precision);
        acb_add(slope, slope, term, precision);
    }
    acb_abs(radius, delta, precision);
    arb_abs(extent, acb_imagref(mid));
    arb_add(extent, extent, radius, precision);
    arb_mul(extent, extent, half, precision);
    arb_exp(envelope, extent, precision);
    if (context) arb_set(c, context->root_l);
    else arb_sqrt(c, l, precision);
    arb_mul(envelope, envelope, c, precision);
    arb_mul(radius, radius, half, precision);
    arb_pow_ui(error, radius, order, precision);
    if (context) arb_set(factorial, context->factorial48);
    else arb_fac_ui(factorial, order, precision);
    arb_div(error, error, factorial, precision);
    arb_mul(error, error, envelope, precision);
    acb_add_error_arb(value, error);
    arb_pow_ui(error, radius, order - 1, precision);
    if (context) arb_set(factorial, context->factorial47);
    else arb_fac_ui(factorial, order - 1, precision);
    arb_div(error, error, factorial, precision);
    arb_mul(error, error, envelope, precision);
    arb_mul(error, error, half, precision);
    acb_add_error_arb(slope, error);
    int ok = acb_is_finite(value) && acb_is_finite(slope);
    arb_clear(c); arb_clear(radius); arb_clear(extent); arb_clear(error);
    arb_clear(factorial); arb_clear(envelope);
    acb_clear(mid); acb_clear(delta); acb_clear(term); acb_clear(q0);
    acb_clear(sine); acb_clear(cosine); acb_clear(previous); acb_clear(coefficient);
    arb_clear(power); _acb_vec_clear(sum, order);
    _acb_vec_clear(near_coefficients, order);
    acb_poly_clear(q); acb_poly_clear(carrier);
    return ok;
}

/* Enclose the entire transform of a finite unit coefficient state over a
 * rectangular complex argument. Exact MPFR coefficients are the finite source;
 * normalization, logarithm, pi and removable carrier limits use ball arithmetic. */
int xc_arb_finite_transform(mpfr_ptr re_lo, mpfr_ptr re_hi, mpfr_ptr im_lo, mpfr_ptr im_hi,
    mpfr_ptr dr_lo, mpfr_ptr dr_hi, mpfr_ptr di_lo, mpfr_ptr di_hi,
    mpfr_srcptr zrl, mpfr_srcptr zrh, mpfr_srcptr zil, mpfr_srcptr zih,
    const char *cutoff, mpfr_srcptr const *coefficients, slong count, slong precision)
{
    acb_t z,q,sinc,derivative,term,value,slope,tmp,b;
    arb_t l,half,pi,norm,c,scale;
    acb_init(z);acb_init(q);acb_init(sinc);acb_init(derivative);acb_init(term);
    acb_init(value);acb_init(slope);acb_init(tmp);acb_init(b);
    arb_init(l);arb_init(half);arb_init(pi);arb_init(norm);arb_init(c);arb_init(scale);
    int ok=0;
    if(count<=0 || count%2==0 || arb_set_str(l,cutoff,precision)!=0 || !arb_is_finite(l))goto cleanup;
    arb_sub_ui(c,l,1,precision);if(!arb_is_positive(c))goto cleanup;
    arb_log(l,l,precision);arb_mul_2exp_si(half,l,-1);
    arb_const_pi(pi,precision);arb_zero(norm);
    for(slong j=0;j<count;j++){arb_set_interval_mpfr(c,coefficients[j],coefficients[j],precision);arb_addmul(norm,c,c,precision);}
    if(!arb_is_finite(norm) || !arb_is_positive(norm))goto cleanup;
    arb_div(scale,l,norm,precision);arb_sqrt(scale,scale,precision);
    arb_set_interval_mpfr(acb_realref(z),zrl,zrh,precision);
    arb_set_interval_mpfr(acb_imagref(z),zil,zih,precision);
    if (mpfr_cmp(zrl, zrh) != 0 || mpfr_cmp(zil, zih) != 0) {
        if (!finite_transform_taylor(value, slope, z, l, half, pi, scale,
                coefficients, count, precision, NULL)) goto cleanup;
        goto output;
    }
    acb_zero(value);acb_zero(slope);
    for(slong j=0;j<count;j++){
        slong mode=j-count/2;
        acb_mul_arb(q,z,half,precision);arb_mul_si(c,pi,mode,precision);arb_add(acb_realref(q),acb_realref(q),c,precision);
        acb_sinc(sinc,q,precision);
        if(acb_contains_zero(q) || near_carrier(q,precision)){
            acb_mul(tmp,q,q,precision);acb_mul_2exp_si(tmp,tmp,-2);acb_neg(tmp,tmp);
            acb_set_si(b,5);acb_mul_2exp_si(b,b,-1);
            acb_hypgeom_0f1(derivative,b,tmp,0,precision);
            acb_mul(derivative,derivative,q,precision);acb_div_ui(derivative,derivative,3,precision);acb_neg(derivative,derivative);
        }else{
            acb_cos(derivative,q,precision);acb_sub(derivative,derivative,sinc,precision);acb_div(derivative,derivative,q,precision);
        }
        arb_set_interval_mpfr(c,coefficients[j],coefficients[j],precision);arb_mul(c,c,scale,precision);
        if(mode%2)arb_neg(c,c);
        acb_mul_arb(term,sinc,c,precision);acb_add(value,value,term,precision);
        arb_mul(c,c,half,precision);acb_mul_arb(term,derivative,c,precision);acb_add(slope,slope,term,precision);
    }
output:
    if(!acb_is_finite(value) || !acb_is_finite(slope))goto cleanup;
    ok=1;
    arb_get_interval_mpfr(re_lo,re_hi,acb_realref(value));arb_get_interval_mpfr(im_lo,im_hi,acb_imagref(value));
    arb_get_interval_mpfr(dr_lo,dr_hi,acb_realref(slope));arb_get_interval_mpfr(di_lo,di_hi,acb_imagref(slope));
cleanup:
    acb_clear(z);acb_clear(q);acb_clear(sinc);acb_clear(derivative);acb_clear(term);acb_clear(value);acb_clear(slope);acb_clear(tmp);acb_clear(b);
    arb_clear(l);arb_clear(half);arb_clear(pi);arb_clear(norm);arb_clear(c);arb_clear(scale);
    return ok;
}

int xc_arb_finite_transform_prepared(mpfr_ptr re_lo, mpfr_ptr re_hi,
    mpfr_ptr im_lo, mpfr_ptr im_hi, mpfr_ptr dr_lo, mpfr_ptr dr_hi,
    mpfr_ptr di_lo, mpfr_ptr di_hi, mpfr_srcptr zrl, mpfr_srcptr zrh,
    mpfr_srcptr zil, mpfr_srcptr zih, const void *opaque)
{
    const xc_arb_finite_transform_context *context = opaque;
    const slong precision = context->precision;
    acb_t z, q, sinc, derivative, term, value, slope, tmp, b;
    acb_init(z); acb_init(q); acb_init(sinc); acb_init(derivative);
    acb_init(term); acb_init(value); acb_init(slope); acb_init(tmp); acb_init(b);
    int ok = 0;
    arb_set_interval_mpfr(acb_realref(z), zrl, zrh, precision);
    arb_set_interval_mpfr(acb_imagref(z), zil, zih, precision);
    if (mpfr_cmp(zrl, zrh) != 0 || mpfr_cmp(zil, zih) != 0) {
        /* scale and source pointers are unused on the prepared path. */
        if (!finite_transform_taylor(value, slope, z, context->l,
                context->half, context->pi, context->half, NULL,
                context->count, precision, context)) goto cleanup;
        goto output;
    }
    acb_zero(value); acb_zero(slope);
    for (slong j = 0; j < context->count; j++) {
        acb_mul_arb(q, z, context->half, precision);
        arb_add(acb_realref(q), acb_realref(q), context->offsets + j, precision);
        acb_sinc(sinc, q, precision);
        if (acb_contains_zero(q) || near_carrier(q, precision)) {
            acb_mul(tmp, q, q, precision); acb_mul_2exp_si(tmp, tmp, -2);
            acb_neg(tmp, tmp); acb_set_si(b, 5); acb_mul_2exp_si(b, b, -1);
            acb_hypgeom_0f1(derivative, b, tmp, 0, precision);
            acb_mul(derivative, derivative, q, precision);
            acb_div_ui(derivative, derivative, 3, precision);
            acb_neg(derivative, derivative);
        } else {
            acb_cos(derivative, q, precision);
            acb_sub(derivative, derivative, sinc, precision);
            acb_div(derivative, derivative, q, precision);
        }
        acb_mul_arb(term, sinc, context->coefficients + j, precision);
        acb_add(value, value, term, precision);
        acb_mul_arb(term, derivative, context->slope_coefficients + j, precision);
        acb_add(slope, slope, term, precision);
    }
output:
    if (!acb_is_finite(value) || !acb_is_finite(slope)) goto cleanup;
    ok = 1;
    arb_get_interval_mpfr(re_lo, re_hi, acb_realref(value));
    arb_get_interval_mpfr(im_lo, im_hi, acb_imagref(value));
    arb_get_interval_mpfr(dr_lo, dr_hi, acb_realref(slope));
    arb_get_interval_mpfr(di_lo, di_hi, acb_imagref(slope));
cleanup:
    acb_clear(z); acb_clear(q); acb_clear(sinc); acb_clear(derivative);
    acb_clear(term); acb_clear(value); acb_clear(slope); acb_clear(tmp); acb_clear(b);
    return ok;
}
void xc_arb_argument(mpfr_ptr lo,mpfr_ptr hi,mpfr_srcptr rl,mpfr_srcptr rh,mpfr_srcptr il,mpfr_srcptr ih,slong precision){
    acb_t z;arb_t angle;acb_init(z);arb_init(angle);
    arb_set_interval_mpfr(acb_realref(z),rl,rh,precision);arb_set_interval_mpfr(acb_imagref(z),il,ih,precision);
    acb_arg(angle,z,precision);arb_get_interval_mpfr(lo,hi,angle);arb_clear(angle);acb_clear(z);
}
