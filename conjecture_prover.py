import numpy as np
import pandas as pd
import segre_veronese_ideal_generation as ig
import linear_conic_intersect as lci
import subspace_manipulation as suma
from scipy import linalg as lg
from plotly import express as px
import timeit

# System imports for filenames
import os
import sys
import glob


# ----------------------- Modular arithmetic utilities -----------------------


def precompute_modular_inverses(p):
    target_power = p - 2
    inds = np.arange(p, dtype=np.uint)
    inv_table = np.ones_like(inds, dtype=np.uint)
    while target_power > 0:
        if target_power % 2 == 1:
            inv_table *= inds
            inv_table %= p
        inds *= inds
        inds %= p
        target_power //= 2
    inv_table[0] = 1
    return inv_table


def find_non_singular_submatrix(matrix, inverse_table):
    rows, cols = matrix.shape
    p = inverse_table.size
    row_order = np.arange(rows)
    det = 1
    deficit = 0
    col_ind = 0
    # print("Initial State")
    # print(matrix)
    while col_ind < cols:
        # print("Matrix")
        # print(matrix)
        row_ind = col_ind - deficit
        first_nonzero = np.argmax(matrix[row_ind:, col_ind] != 0) + row_ind
        if matrix[first_nonzero, col_ind] == 0:
            # print("Final state")
            # print(matrix)
            return 0, None, col_ind

        if first_nonzero != row_ind:
            # Swap rows
            # print("Rows swapped")
            # print(matrix)
            matrix[[row_ind, first_nonzero], :] = matrix[[first_nonzero, row_ind], :]
            row_order[[row_ind, first_nonzero]] = row_order[[first_nonzero, row_ind]]
        det *= matrix[row_ind, col_ind]
        det %= p
        invs = inverse_table[matrix[row_ind:, col_ind]]
        matrix[row_ind:, col_ind:] = matrix[row_ind:, col_ind:] * invs[:, np.newaxis]

        # print("Normalized")
        # print(matrix % p)
        first_row = p - (matrix[row_ind, col_ind:] % p)
        nnz_rows = (matrix[:, col_ind] % p) != 0
        nnz_rows[: (row_ind + 1)] = False
        matrix[nnz_rows, col_ind:] += first_row[np.newaxis, :]
        matrix[row_ind:, col_ind:] %= p
        # print("Eliminated")
        # print(matrix)
        col_ind += 1
    return det, row_order[cols:], cols


# ----------------------- Test case organizers -----------------------
def max_planted_solutions(n1, n2):
    return np.sqrt(0.5 * n1 * (n1 - 1) * n2 * (n2 - 1) + 0.25) + 0.5


# Get the maximum subspace dimension supportable by conjecture given s planted
# rank-1 n1 x n2 matrices
def max_dimensions(n1, n2, s):
    return np.sqrt(0.5 * n1 * (n1 - 1) * n2 * (n2 - 1) + 0.25 + 2 * s) - 0.5


def max_planted_symmetric_solutions(n):
    return np.sqrt((n + 1) * n**2 * (n - 1) / 6 + 0.25) + 0.5


def max_symmetric_dimensions(n, s):
    return np.sqrt((n + 1) * n**2 * (n - 1) / 6 + 0.25 + 2 * s) - 0.5


# Returns all (m,n) pairs such that m*n/sqrt(2) < upper_bound
# and the associated maximum subspace dimension for an m-by-n matrix
def list_shapes(upper_bound):
    ms = []
    ns = []
    m_max = int(np.sqrt(np.sqrt(2) * upper_bound))
    for m in range(2, m_max + 1):
        n_max = int(np.sqrt(2) * upper_bound / m)
        n_seg = np.arange(m, n_max + 1)
        m_seg = np.ones_like(n_seg) * m
        ns.append(n_seg)
        ms.append(m_seg)
    ns = np.hstack(ns)
    ms = np.hstack(ms)
    rs = max_planted_solutions(ms, ns)
    ord = np.argsort(rs)
    vals = np.vstack((ms, ns, rs.astype(int)))[:, ord]
    return vals.transpose()


# Returns all values of m such that n^2/sqrt(6) < upper_bound
# and the associated maximum subspace dimension for symmetric m-by-m matrix
def list_sym_shapes(upper_bound):
    max_m = int(np.sqrt(np.sqrt(6) * upper_bound))
    arr = np.zeros((max_m - 1, 2), dtype=np.uint)
    arr[:, 0] = np.arange(2, max_m + 1)
    arr[:, 1] = max_planted_symmetric_solutions(arr[:, 0])
    return arr


# Returns all (m,n,R,s) parameters that meet the conditions of the conjecture
# with m*n/sqrt(2) < upper_bound and 'R' as large as possible for a given 's'
def list_subshapes(upper_bound):
    blocks = []
    shapes = list_shapes(upper_bound)
    for m, n, R_max in shapes:
        block = np.zeros((R_max + 1, 4), dtype=np.uint)
        block[:, 0] = m
        block[:, 1] = n
        block[:, 3] = np.arange(0, R_max + 1, dtype=np.uint)
        block[:, 2] = max_dimensions(m, n, block[:, 3]).astype(np.uint)
        blocks.append(block)
    all_shapes = np.vstack(blocks)
    return all_shapes


# Returns all (m,R,s) parameters that meet the conditions of the conjecture
# with m^2/sqrt(6) < upper_bound and 'R' as large as possible for a given 's'
def list_sym_subshapes(max_dimension):
    blocks = []
    sym_shapes = list_sym_shapes(max_dimension)
    for m, R_max in sym_shapes:
        block = np.zeros((R_max + 1, 3), dtype=np.uint)
        block[:, 0] = m
        block[:, 2] = np.arange(0, R_max + 1, dtype=np.uint)
        block[:, 1] = max_symmetric_dimensions(m, block[:, 2])
        blocks.append(block)
    all_shapes = np.vstack(blocks)
    return all_shapes


# Returns all (m,n,R,s) parameters with m*n/sqrt(2) < upper_bound
# so that 'R' is one larger than what is believed possible for a given 's' by
# the conjecture as long as 'R' is not to large as to be un-identifiable
def list_overbound_subshapes(max_dimension):
    blocks = []
    tests = list_shapes(max_dimension)
    for m, n, R_max in tests:
        block = np.zeros((R_max + 2, 4), dtype=np.uint)
        block[:, 0] = m
        block[:, 1] = n
        block[:, 3] = np.arange(0, R_max + 2, dtype=np.uint)
        block[-1, 3] = R_max + 1
        block[:, 2] = max_dimensions(m, n, block[:, 3]).astype(np.uint) + 1
        blocks.append(block)
    all_shapes = np.vstack(blocks)
    # Filter by which shapes are within the identifiability bound
    # (eliminates some (2,n) edge-cases mainly)
    identifiable = all_shapes[:, 2] <= (all_shapes[:, 0] - 1) * (all_shapes[:, 1] - 1)
    return all_shapes[identifiable, :]


# Returns all (m,R,s) parameters with m^2/sqrt(6) < upper_bound
# so that 'R' is one larger than what is believed possible for a given 's' by
# the conjecture
def list_overbound_sym_subshapes(upper_bound):
    blocks = []
    tests = list_sym_shapes(upper_bound)
    for m, R_max in tests:
        block = np.zeros((R_max + 2, 3), dtype=np.uint)
        block[:, 0] = m
        block[:, 2] = np.arange(0, R_max + 2, dtype=np.uint)
        block[-1, 2] = R_max + 1
        block[:, 1] = max_symmetric_dimensions(m, block[:, 2]).astype(np.uint) + 1
        blocks.append(block)
    all_shapes = np.vstack(blocks)
    return all_shapes[all_shapes[:, 1] <= upper_bound]


# ----------------------- Certification procedures -----------------------
def find_certified_subspace_seeds(
    param_list,
    mults,
    modulus,
    record_name,
    log_name,
    overwrite=False,
    max_gen_attempts=10,
):
    log = open(log_name, "w" if overwrite else "a")
    log.write("---------------------------------------------------------\n")
    log.write("Starting new session.\n")
    index = 0
    if not overwrite:
        try:
            current_data = pd.read_csv(record_name)
            index = current_data.shape[0]
        except FileNotFoundError:
            log.write(
                "Failed to find "
                + record_name
                + " with previous records so starting new file.\n"
            )

    inv_table = precompute_modular_inverses(modulus)
    num_shapes = param_list.shape[0]
    frame_headers = [
        "m",
        "n",
        "R",
        "s",
        "seed",
        "det_mod" + str(modulus),
        "rm_rows",
    ]
    while index < num_shapes:
        start_time = timeit.default_timer()

        dims = param_list[index, :-2]
        R = param_list[index, -2]
        s = param_list[index, -1]

        instance_string = "Certifying seed, case {}".format(
            tuple(int(item) for item in param_list[index, :])
        )
        log.write(instance_string + "\n")
        print(instance_string)
        data = {"m": dims[0], "R": R, "s": s}
        if dims.size > 1:
            data.update({"n": dims[1]})
        try:
            seed, det, excluded_rows = generate_witness(
                dims,
                mults,
                R,
                s,
                inv_table,
                max_attempts=max_gen_attempts,
                log_file=log,
            )
            data.update(
                {"seed": seed, "det_mod" + str(modulus): det, "rm_rows": excluded_rows}
            )
            log.write("  Trials required: " + str(seed + 1) + "\n")
        except np.exceptions.TooHardError as err:
            log.write("  " + str(err) + "\n")

        row = pd.DataFrame([data], columns=frame_headers)
        row.to_csv(
            record_name,
            mode="w" if (index == 0) and overwrite else "a",
            header=index == 0,
            index=False,
        )
        end_time = timeit.default_timer()
        log.write("  Elapsed time: %f\n" % (end_time - start_time))
        index += 1


def generate_witness(dims, mults, R, s, inv_table, log_file=None, max_attempts=10):
    p = inv_table.size
    for seed in range(max_attempts):
        basis = generate_modular_subspace(dims, mults, R, s, inv_table, seed)
        b_det, _, _ = find_non_singular_submatrix(basis % p, inverse_table=inv_table)
        if b_det == 0:
            log_file.write("  Random subspaces has dependent vectors, regenerating.\n")
            continue
        det, excluded_rows = m_matrix_determinant(basis % p, dims, mults, s, inv_table)
        if det != 0:
            return seed, det, excluded_rows
    raise np.exceptions.TooHardError(
        "Failed to find witness after " + str(max_attempts) + " random initializations."
    )


# DO NOT TOUCH, altering procedure will invalidate ALL CERTIFIED SEEDS
def generate_modular_subspace(dims, mults, R, s, inv_table, seed):
    # Seeding random number generation ensures reproducibility, if this function
    # is called with same parameters it should produce the same matrix
    rng = np.random.default_rng(seed)
    N = ig.basis_size_for(dims, mults)
    planted = np.zeros((N, 0), dtype=np.int32)
    extras = np.zeros((N, 0), dtype=np.int32)
    if s < R:
        extras = uniform_sample_Fp_vectors(N, R - s, inv_table.size, rng)
        # extras = pairwise_lin_indep_vector_set(N, R - s, inv_table, rng)
        # print("Extras")
        # print(extras)
    if s > 0:
        factors = []
        for dim in dims:
            factor = pairwise_lin_indep_vector_set(dim, s, inv_table, rng)
            # print("Factor")
            # print(factor)
            factors.append(factor)
        planted = suma.khatri_rhao_products(factors, mults)
    return np.hstack((planted, extras))


# DO NOT TOUCH, altering procedure will invalidate ALL CERTIFIED SEEDS currently
# recorded
# A discretization of Guassian random vectors centered at (0,0,...,0)
def uniform_sample_Fp_vectors(n, num_vecs, p, rng):
    arr = rng.integers(0, p, size=(n, num_vecs), dtype=np.int32)
    return arr


# Unused, but maybe for future experiments
def sparse_k_sampler(k):
    def sparse_sample_Fp_vectors(n, num_vecs, p, rng):
        vals_per_col = k
        arr = np.zeros((n, num_vecs), dtype=np.int32)
        vals = rng.integers(0, p, size=(vals_per_col, num_vecs), dtype=np.int32)
        if n <= vals_per_col:
            return vals[:n, :]
        for i in range(num_vecs):
            row_ind = rng.choice(n, size=(vals_per_col), replace=False)
            col_ind = i * np.ones((vals_per_col,), dtype=int)
            # print(row_ind, col_ind)
            arr[row_ind, col_ind] = vals[:, i]
        return arr

    return sparse_sample_Fp_vectors


# DO NOT TOUCH, altering procedure will invalidate all certified seed
# Pairwise linear independent modulo p is a necessary condition to certify
# the M matrix determinant is non-zero modulo p so ensure this holds so we
# don't do useless computation later on
def pairwise_lin_indep_vector_set(
    n, num_vecs, inv_table, rng, sampler=uniform_sample_Fp_vectors
):
    # if num_vecs == 0:
    p = inv_table.size
    if np.log(num_vecs) > n * np.log(p) - np.log(p - 1) + np.log(1 - 1 / p**n):
        raise ValueError(
            "There are fewer than {} pairwise linearly independent vectors over F^{}_{}.".format(
                num_vecs, n, p
            )
        )
    num_pwli = 0
    bank = None

    # Add more vectors to the bank while we do not have a pairwise linearly
    # independent set
    while num_pwli < num_vecs:
        # Adds vectors in batches for efficiency, worst case expect logarithmic
        # number of iterations since this is a stamp collector-like problem
        bank = (
            sampler(n, num_vecs, p, rng)
            if bank is None
            else np.hstack([bank, sampler(n, num_vecs, p, rng)])
        )
        # Find a canonical form for each vector by normalizing so first non-zero
        # entry is 1
        firstnnz = np.argmax(bank != 0, axis=0)
        canon = bank * inv_table[bank[firstnnz, np.arange(bank.shape[1])] % p] % p
        # Determine a unique set of canonical vectors and their locations in the
        # original bank
        uni, inds = np.unique(canon, return_index=True, axis=1)
        # If the first vector is the zero vector, exclude it
        start = 1 if np.all(uni[:, 0] == 0) else 0
        num_pwli = uni.shape[1] - start
    # Sorting ensures we take vectors in order of generation and not in the
    # lexicographic ordering induced by the 'unique' function
    sel = np.sort(inds[start : (num_vecs + start)])
    return bank[:, sel]


def m_matrix_determinant(basis, dims, mults, s, inv_table):
    p = inv_table.size
    lift = suma.symmetric_lift(basis[:, ::-1] % p, 2)[:, ::-1] % p
    # print("Basis is:", basis.dtype)
    exclude = np.cumsum(np.arange(1, s + 1, dtype=np.uint)) - 1
    # print(exclude)
    partial_lift = np.delete(lift, exclude, 1)
    det_mat = ig.sv_poly_matrix(dims, mults)
    m_matrix = ((det_mat @ partial_lift) % p).astype(np.uint)
    # print(det_mat)
    # print(exclude)
    # print((det_mat @ lift) % p)
    # print("M matrix")
    # print(m_matrix)

    # cols = m_matrix.shape[1]
    # m_matrix = m_matrix[:cols, :]
    det, excluded_rows, _ = find_non_singular_submatrix(m_matrix, inv_table)
    return det, excluded_rows


# ----------------------- Numerical test procedures -----------------------
def test_numerical_recovery(
    param_list,
    mults,
    record_name,
    log_name,
    overwrite=False,
    max_attempts=10,
    max_rounding=3,
    decomp_tol=1e-6,
    sing_tol=1e-9,
    error_tol=1e-12,
):
    log = open(log_name, "w" if overwrite else "a")
    log.write("---------------------------------------------------------\n")
    log.write("Starting new session.\n")
    index = 0
    if not overwrite:
        try:
            current_data = pd.read_csv(record_name)
            index = current_data.shape[0]
        except FileNotFoundError:
            log.write(
                "Failed to find "
                + record_name
                + " with previous records so starting new file.\n"
            )

    num_shapes = param_list.shape[0]
    frame_headers = [
        "m",
        "n",
        "R",
        "s",
        "seed",
        "ker_dim",
        "s_val",
        "decomp_error",
        "w",
    ]
    while index < num_shapes:
        start_time = timeit.default_timer()
        dims = param_list[index, :-2]
        R = param_list[index, -2]
        s = param_list[index, -1]

        instance_string = "Running numerics, case {}".format(
            tuple(int(item) for item in param_list[index, :])
        )
        log.write(instance_string + "\n")
        print(instance_string)
        data = {"m": dims[0], "R": R, "s": s, "seed": 0}
        if dims.size > 1:
            data.update({"n": dims[1]})
        try:
            seed, kernel_dim, sing_val, decomp_error, w = verify_floating_pt_recovery(
                dims,
                mults,
                R,
                s,
                max_attempts=max_attempts,
                max_rounding=max_rounding,
                decomp_tol=decomp_tol,
                sing_tol=sing_tol,
                error_tol=error_tol,
                log_file=log,
            )
            data.update(
                {
                    "seed": seed,
                    "ker_dim": kernel_dim,
                    "s_val": sing_val,
                    "decomp_error": decomp_error,
                    "w": w,
                }
            )
        except np.exceptions.TooHardError as err:
            log.write("  " + err.args[0] + "\n")

        row = pd.DataFrame([data], columns=frame_headers)
        row.to_csv(
            record_name,
            mode="w" if (index == 0) and overwrite else "a",
            header=index == 0,
            index=False,
        )
        end_time = timeit.default_timer()
        log.write("  Elapsed time: %f\n" % (end_time - start_time))
        index += 1


def verify_floating_pt_recovery(
    dims,
    mults,
    R,
    s,
    max_attempts=10,
    max_rounding=3,
    decomp_tol=1e-6,
    sing_tol=1e-9,
    error_tol=1e-12,
    log_file=None,
):
    for seed in range(max(max_attempts, 1)):
        real_basis = generate_real_subspace(dims, mults, R, s, seed)
        orth_basis = lg.orth(real_basis)
        try:
            recovered, kernel_dim, sing_value, decomp_error = find_planted_tensors(
                orth_basis,
                dims,
                mults,
                max_rounding,
                sing_tol,
                decomp_tol,
                log_file=None,
            )
            if (kernel_dim > 0) and (s > 0):
                w = 1 - np.min(suma.similarity_between(recovered, real_basis[:, :s]))
                if w < error_tol:
                    return seed, kernel_dim, sing_value, decomp_error, w
            return seed, kernel_dim, sing_value, decomp_error, np.nan
        except (np.linalg.LinAlgError, np.exceptions.TooHardError) as err:
            log_file.write("  " + str(err) + "\n")
    raise np.exceptions.TooHardError(
        "Failed to verify numerical example after "
        + str(max_attempts)
        + " random initializations."
    )


def find_planted_tensors(
    basis,
    dims,
    mults,
    max_rounding,
    sing_tol,
    decomp_tol,
    log_file=None,
):
    R = basis.shape[1]
    lift = suma.symmetric_lift(basis, 2)
    minors_matrix = ig.sv_poly_matrix(dims, mults) @ lift
    sing_val = 0
    if minors_matrix.shape[1] == 1:
        sing_val = np.linalg.norm(minors_matrix, ord="fro")
        if sing_val < sing_tol:
            kernel_basis = np.array([[1]])
            sing_val = np.nan
        else:
            return None, 0, sing_val, np.nan
    else:
        _, sing_vals, vh = lg.svd(minors_matrix, overwrite_a=True)
        ind = np.sum(sing_vals > sing_tol)
        sing_val = sing_vals[ind - 1] if ind < sing_vals.size else sing_vals[-1]
        weights = suma.symmetric_weights(R, 2)
        kernel_basis = vh[ind:, :].transpose() / weights[:, None]
    kernel_dim = kernel_basis.shape[1]
    if kernel_dim <= 0:
        return None, 0, sing_val, np.nan
    ktensor_dims = [R, kernel_dim]
    ktensor_mults = [2, 1]
    # mode_expand = [[1, 1, 0], [0, 0, 1]]
    for rounding_seed in range(max_rounding):
        # Seed here determine the random vectors for contracting the tensor
        rng = np.random.default_rng(rounding_seed)
        # Jennrich's algorithm recovers the coefficients on basis vectors
        # to produce planted solutions
        factors, diagnostics, _ = lci.jennrich_partial_decomp(
            kernel_basis.flatten(),
            ktensor_dims,
            ktensor_mults,
            rng=rng,
            atol=sing_tol,
            expected_rank=kernel_dim,
        )
        error = diagnostics[0]
        recovered = basis @ factors[0]
        if error < decomp_tol:
            return recovered, kernel_dim, sing_val, error
    raise np.exceptions.TooHardError("Jennrich's failed to decompose tensor.")


# DO NOT TOUCH, altering procedure will invalidate numerical results
def generate_real_subspace(dims, mults, R, s, seed):
    # Seeding random number generation ensures reproducibility, if this function
    # is called with same parameters it should produce the same matrix
    rng = np.random.default_rng(seed)
    N = ig.basis_size_for(dims, mults)
    planted = np.zeros((N, 0))
    extras = np.zeros((N, 0))
    if s < R:
        extras = rng.normal(size=(N, R - s))
        # extras = pairwise_lin_indep_vector_set(N, R - s, inv_table, rng)
        # print("Extras")
        # print(extras)
    if s > 0:
        factors = []
        for dim in dims:
            factor = rng.normal(size=(dim, s))
            # print("Factor")
            # print(factor)
            factors.append(factor)
        planted = suma.khatri_rhao_products(factors, mults)
    return np.hstack((planted, extras))


# ----------------------- Test batch hook-ins -----------------------


def get_filename_defaults(
    max_dimension, modulus, batch_name, test_type, min_dimension=None
):
    if min_dimension is None:
        base_filename = "{}_b{}_p{}".format(batch_name, max_dimension, modulus)
    else:
        base_filename = "{}_b{}-{}_p{}".format(
            batch_name, min_dimension, max_dimension, modulus
        )
    recordname = os.path.join(".", "data", test_type, base_filename + ".csv")
    logname = os.path.join(
        ".", "data", "logs", "log_" + test_type + "_" + base_filename + ".txt"
    )
    # recordname = test_type + "/" + base_filename + ".csv"
    # logname = "logs/" + test_type + "_" + base_filename + ".txt"
    return recordname, logname


def get_params_and_multiplicities(test_class, upper_bound, lower_bound=None):
    mults = [1, 1]
    is_sym = False
    if test_class == "all":
        param_set = list_subshapes(upper_bound)
    elif test_class == "cpd":
        partial_params = list_shapes(upper_bound)
        param_set = np.zeros((partial_params.shape[0], 4), dtype=np.int32)
        param_set[:, 0:3] = partial_params
        param_set[:, 3] = param_set[:, 2]
    elif test_class == "null":
        partial_params = list_shapes(upper_bound)
        param_set = np.zeros((partial_params.shape[0], 4), dtype=np.int32)
        param_set[:, 0:3] = partial_params
        param_set[:, 2] -= 1
    elif test_class == "all_sym":
        param_set = list_sym_subshapes(upper_bound)
        mults = [2]
        is_sym = True
    elif test_class == "cpd_sym":
        partial_params = list_sym_shapes(upper_bound)
        param_set = np.zeros((partial_params.shape[0], 3), dtype=np.int32)
        param_set[:, 0:2] = partial_params
        param_set[:, 2] = param_set[:, 1]
        mults = [2]
        is_sym = True
    elif test_class == "null_sym":
        partial_params = list_sym_shapes(upper_bound)
        param_set = np.zeros((partial_params.shape[0], 3), dtype=np.int32)
        param_set[:, 0:2] = partial_params
        param_set[:, 1] -= 1
        mults = [2]
        is_sym = True
    elif test_class == "overbound":
        param_set = list_overbound_subshapes(upper_bound)
    elif test_class == "overbound_sym":
        param_set = list_overbound_sym_subshapes(upper_bound)
        mults = [2]
        is_sym = True
    else:
        raise ValueError("Test class '{}' not recognized.".format(test_class))

    if lower_bound is not None:
        if is_sym:
            sel = param_set[:, 0] ** 2 / np.sqrt(6) >= lower_bound
            param_set = param_set[sel, :]
        else:
            sel = param_set[:, 0] * param_set[:, 1] / np.sqrt(2) >= lower_bound
            param_set = param_set[sel, :]
    return param_set, mults


def generate_certificates(
    test_class,
    upper_bound,
    modulus,
    lower_bound=None,
    overwrite=False,
    max_gen_attempts=10,
):
    if test_class.split("_")[0] == "overbound":
        raise ValueError(
            "Impossible to produce certificates for any cases over the bound."
        )

    record_name, log_name = get_filename_defaults(
        upper_bound, modulus, test_class, "certificates", min_dimension=lower_bound
    )
    # print(record_name)
    # print(log_name)
    param_set, mults = get_params_and_multiplicities(
        test_class, upper_bound, lower_bound
    )
    # print(param_set)

    find_certified_subspace_seeds(
        param_set,
        mults,
        modulus,
        record_name,
        log_name,
        overwrite=overwrite,
        max_gen_attempts=max_gen_attempts,
    )


def run_numerical_recovery(
    test_class,
    upper_bound,
    overwrite=False,
    lower_bound=None,
    max_attempts=10,
    max_rounding=3,
    decomp_tol=1e-6,
    sing_tol=1e-9,
    error_tol=1e-12,
):
    if test_class.split("_")[0] == "overbound":
        decomp_tol = np.inf
        error_tol = np.inf

    param_set, mults = get_params_and_multiplicities(
        test_class, upper_bound, lower_bound
    )

    numerical_record, numerical_log = get_filename_defaults(
        upper_bound,
        modulus,
        test_class,
        "numerical",
        min_dimension=lower_bound,
    )

    test_numerical_recovery(
        param_set,
        mults,
        numerical_record,
        numerical_log,
        max_attempts=max_attempts,
        max_rounding=max_rounding,
        decomp_tol=decomp_tol,
        sing_tol=sing_tol,
        error_tol=error_tol,
        overwrite=overwrite,
    )


def generate_labeled_table(save_name, m_max, n_max, bounds):
    filepath = os.path.join(".", "data", save_name + ".txt")
    file = open(filepath, "w")

    # Make a header
    file.write("$m \\backslash n$")
    for n in range(2, n_max + 1):
        file.write(" & " + str(n))
    file.write(" \\\\ \\hline \n")
    for m in range(2, m_max + 1):
        file.write(str(m))
        for n in range(2, n_max + 1):
            R = int(max_planted_solutions(m, n))
            t = n * m / np.sqrt(2)
            if n > m:
                file.write(" & \\color{lightgray}" + str(R))
            elif t <= bounds[0]:
                file.write(" & \\cellcolor{myblue}" + str(R))
            elif t <= bounds[1]:
                file.write(" & \\cellcolor{mypurple}" + str(R))
            elif t <= bounds[2]:
                file.write(" & \\cellcolor{mymagenta}" + str(R))
            elif t <= bounds[3]:
                file.write(" & \\cellcolor{myorange}" + str(R))
            else:
                file.write(" & " + str(R))
        if m != m_max:
            file.write(" \\\\\n")
    file.close()


# ---------------- Choice of parameters ----------------
modulus = 997

certified_all_bound = 80
certified_part_bound = 120

numerical_all_bound = 80
numerical_part_bound = 165
numerical_over_bound = 60

all_sym_bound = 90
part_sym_bound = 190


def run_default_tests():
    # ---------------- Write latex table for test params ----------------
    generate_labeled_table(
        "proof_and_numerics_table",
        40,
        15,
        bounds=[
            certified_all_bound,
            numerical_all_bound,
            certified_part_bound,
            numerical_part_bound,
        ],
    )

    # ---------------- Main tests to run ----------------
    generate_certificates("all", certified_all_bound, modulus)
    generate_certificates(
        "cpd", certified_part_bound, modulus, lower_bound=certified_all_bound + 1
    )
    generate_certificates(
        "null", certified_part_bound, modulus, lower_bound=certified_all_bound + 1
    )

    run_numerical_recovery("all", numerical_all_bound)
    run_numerical_recovery(
        "cpd", numerical_part_bound, lower_bound=numerical_all_bound + 1
    )
    run_numerical_recovery(
        "null", numerical_part_bound, lower_bound=numerical_all_bound + 1
    )
    run_numerical_recovery("overbound", numerical_over_bound)

    # ---------------- Symmetric tests to run ----------------

    generate_certificates("all_sym", all_sym_bound, modulus)
    generate_certificates(
        "cpd_sym", part_sym_bound, modulus, lower_bound=all_sym_bound + 1
    )
    generate_certificates(
        "null_sym", part_sym_bound, modulus, lower_bound=all_sym_bound + 1
    )

    run_numerical_recovery("all_sym", all_sym_bound)
    run_numerical_recovery("cpd_sym", part_sym_bound, lower_bound=all_sym_bound + 1)
    run_numerical_recovery("null_sym", part_sym_bound, lower_bound=all_sym_bound + 1)
    run_numerical_recovery("overbound_sym", all_sym_bound)


def analyze_data():
    p = 997
    print("----- Original case ------")
    reg_cert_files = [
        "all_b60_p997.csv",
        "cpd_b61-120_p997.csv",
        "null_b61-120_p997.csv",
    ]

    reg_cert_data = [
        pd.read_csv(os.path.join("data", "certificates", file))
        for file in reg_cert_files
    ]
    certificates = pd.concat(reg_cert_data, ignore_index=True)
    print("Certifications")
    print("  Seeds")
    seed_vals, seed_counts = np.unique(certificates["seed"], return_counts=True)
    print("    Seed values:", seed_vals)
    print("    Seed counts:", seed_counts)

    reg_num_files = [
        "all_b80_p997.csv",
        "cpd_b81-165_p997.csv",
        "null_b81-165_p997.csv",
    ]
    reg_numerical_data = [
        pd.read_csv(os.path.join("data", "numerical", file)) for file in reg_num_files
    ]
    numerics = pd.concat(reg_numerical_data, ignore_index=True)
    print("Numerical results")
    print("  Below conjectured bound")
    print("    Minimum s_val:", np.min(numerics["s_val"]))
    print("    Max decomp error:", np.max(numerics["decomp_error"]))
    print("    Max matching error:", np.max(numerics["w"]))
    overbound = pd.read_csv(os.path.join("data", "numerical", "overbound_b60_p997.csv"))
    print("  Above conjectured bound")
    print("    Minimum s_val:", np.min(overbound["s_val"]))
    extra_elements = overbound["ker_dim"] - overbound["s"]

    print("    Minimal extra elements:", np.min(extra_elements))
    print("    Min decomp error:", np.min(overbound["decomp_error"]))
    print("    Min matching error:", np.min(overbound["w"]))
    # Check predicted size of kernel
    m = overbound["m"]
    n = overbound["n"]
    R = overbound["R"]
    overbound["pred_kernel"] = R * (R + 1) // 2 - m * (m - 1) * n * (n - 1) // 4
    kernel_prediction_excess = overbound["ker_dim"] - overbound["pred_kernel"]
    print(
        "    Kernel deviations from prediction:", np.sum(kernel_prediction_excess != 0)
    )

    print("\n----- Symmetric case ------")
    sym_cert_files = [
        "all_sym_b90_p997.csv",
        "cpd_sym_b91-190_p997.csv",
        "null_sym_b91-190_p997.csv",
    ]
    sym_cert_data = [
        pd.read_csv(os.path.join("data", "certificates", file))
        for file in reg_cert_files
    ]
    sym_certificates = pd.concat(sym_cert_data, ignore_index=True)
    print("Certifications")
    print("  Seeds")
    seed_vals, seed_counts = np.unique(sym_certificates["seed"], return_counts=True)
    print("    Seed values:", seed_vals)
    print("    Seed counts:", seed_counts)

    reg_num_files = [
        "all_sym_b90_p997.csv",
        "cpd_sym_b91-190_p997.csv",
        "null_sym_b91-190_p997.csv",
    ]
    sym_numerical_data = [
        pd.read_csv(os.path.join("data", "numerical", file)) for file in reg_num_files
    ]
    numerics = pd.concat(sym_numerical_data, ignore_index=True)
    print("Numerical results")
    print("  Below conjectured bound")
    print("    Minimum s_val:", np.min(numerics["s_val"]))
    print("    Max decomp error:", np.max(numerics["decomp_error"]))
    print("    Max matching error:", np.max(numerics["w"]))
    overbound = pd.read_csv(
        os.path.join("data", "numerical", "overbound_sym_b90_p997.csv")
    )
    print("  Above conjectured bound")
    print("    Minimum s_val:", np.min(overbound["s_val"]))
    extra_elements = overbound["ker_dim"] - overbound["s"]

    print("    Minimal extra elements:", np.min(extra_elements))
    print("    Min decomp error:", np.min(overbound["decomp_error"]))
    print("    Min matching error:", np.min(overbound["w"]))
    # Check predicted size of kernel
    m = overbound["m"]
    R = overbound["R"]
    overbound["pred_kernel"] = R * (R + 1) // 2 - (m + 1) * m**2 * (m - 1) // 12
    kernel_prediction_excess = overbound["ker_dim"] - overbound["pred_kernel"]
    print(
        "    Kernel deviations from prediction:", np.sum(kernel_prediction_excess != 0)
    )


if __name__ == "__main__":
    analyze_data()
