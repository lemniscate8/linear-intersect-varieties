import itertools
import numpy as np
from scipy.special import binom, factorial
from scipy.linalg import khatri_rao
from scipy import sparse
import segre_veronese_ideal_generation as ig

rng = np.random.default_rng()


# Helper function for computing general Khatri-Rhao products including no
# redundancy for higher multiplicies
def khatri_rhao_products(factors, mults=None):
    factors_type = factors[0].dtype
    if (mults is None) or (np.all(np.array(mults, dtype=int) == 1)):
        tensors = np.ones((1, factors[0].shape[1]), dtype=factors_type)
        for factor_mat in factors:
            tensors = khatri_rao(tensors, factor_mat)
        return tensors
    bases = [
        symmetric_khatri_rhao(factor, mult) for factor, mult in zip(factors, mults)
    ]
    return khatri_rhao_products(bases)


# Compute repeated Khatri-Rhao products but only keep elements in symmetric
# tensors that are not redundant
def symmetric_khatri_rhao(basis, k):
    if k == 0:
        return np.ones((1, basis.shape[1]))
    elif k == 1:
        return basis
    nrow = basis.shape[0]
    lrow = int(binom(nrow + k - 1, k))
    row_indices = np.fromiter(
        itertools.combinations_with_replacement(range(nrow), k),
        dtype=np.dtype((int, k)),
        count=lrow,
    )
    prod = 1
    for ind in range(k):
        prod = basis[row_indices[:, ind], :] * prod
    return prod


# Compute repeated tensor products or symmetric tensor products (vee) while
# saving in compact form
def tensor_products(matrices, mults=None):
    if (mults is None) or (np.all(np.array(mults, dtype=int) == 1)):
        tensor = 1
        for factor_mat in matrices:
            if sparse.issparse(factor_mat) or sparse.issparse(tensor):
                tensor = sparse.kron(tensor, factor_mat)
            else:
                tensor = np.kron(tensor, factor_mat)
        return tensor
    sym_lifts = [symmetric_lift(matrix, mult) for matrix, mult in zip(matrices, mults)]
    return tensor_products(sym_lifts)


# Computes the symmetric lift of a basis
# Note: the basis must either be a dense array or sparse array, sparse _matrices_
# will cause an error
def symmetric_lift(basis, k):
    bool_basis = basis.dtype == np.bool

    if k == 1:
        return basis
    nrow, ncol = basis.shape
    lrow = int(binom(nrow + k - 1, k))
    lcol = int(binom(ncol + k - 1, k))
    if sparse.issparse(basis):
        lifted_basis = sparse.csr_array((lrow, lcol), dtype=basis.dtype)
    else:
        lifted_basis = np.zeros(shape=(lrow, lcol), dtype=basis.dtype)
    row_iter = np.fromiter(
        itertools.combinations_with_replacement(range(nrow), k),
        dtype=np.dtype((int, k)),
        count=lrow,
    )
    col_iter = np.fromiter(
        itertools.combinations_with_replacement(range(ncol), k),
        dtype=np.dtype((int, k)),
        count=lcol,
    )
    # print(col_iter)
    # print(row_iter)
    k_fact = 0
    for perm in itertools.permutations(range(k)):
        prod = 1
        for index in range(k):
            # print(row_iter[:, index, None])
            # print(col_iter[:, perm[index]])
            row_col_selection = basis[
                row_iter[:, index, None], col_iter[:, perm[index]]
            ]
            if bool_basis:
                prod = row_col_selection & prod
            else:
                prod = row_col_selection * prod
        if bool_basis:
            lifted_basis ^= prod
        else:
            lifted_basis += prod
        k_fact += 1
    if bool_basis or np.issubdtype(basis.dtype, np.integer):
        return lifted_basis
    return lifted_basis / k_fact


# Weights for switching from canonical Veronese embedding to a Segre embedding
def symmetric_weights(basis_dim, k):
    num_entries = int(binom(basis_dim + k - 1, k))
    col_iter = itertools.combinations_with_replacement(range(basis_dim), k)
    nunique = lambda arr_like: np.unique(arr_like).size
    weights = np.fromiter(
        map(nunique, col_iter),
        dtype=int,
        count=num_entries,
    )
    return factorial(weights)


# Produces a tall or square matrix that is a look up table fora balanced
# flattening of a symmetric tensor
def symmetric_flattening_lut(flat_dim, mult):
    if mult == 1:
        return np.arange(flat_dim)
    col_mult = mult // 2
    row_mult = mult - col_mult
    row_dim = int(binom(flat_dim + row_mult - 1, row_mult))
    col_dim = int(binom(flat_dim + col_mult - 1, col_mult))
    lut = {}
    for i, tup in enumerate(
        itertools.combinations_with_replacement(range(flat_dim), mult)
    ):
        lut[tup] = i

    row_iter = itertools.combinations_with_replacement(range(flat_dim), row_mult)
    col_iter = itertools.combinations_with_replacement(range(flat_dim), col_mult)

    index_lookup = lambda tups: lut[tuple(sorted(tups[0] + tups[1]))]

    lut_matrix = np.fromiter(
        map(index_lookup, itertools.product(row_iter, col_iter)),
        dtype=int,
        count=row_dim * col_dim,
    ).reshape((row_dim, col_dim))
    return lut_matrix


# Generalization of symmetric flattening to work for any pattern of modes
def tensor_flattening_pattern(dims, mults, row_modes):
    mult1 = np.array(row_modes, dtype=int)
    mult2 = np.subtract(mults, row_modes)
    if np.any(mult2 < 0):
        raise Exception("Extractions must be less than multiplicity.")
    lookup = {}
    for index, code in enumerate(ig.basis_generator(dims, mults)):
        lookup[code] = index
    col_iter = ig.basis_generator(dims, mult1)
    col_dim = ig.basis_size_for(dims, mult1)
    row_iter = ig.basis_generator(dims, mult2)
    row_dim = ig.basis_size_for(dims, mult2)
    flatten = lambda tup: lookup[
        tuple(tuple(sorted(a + b)) for a, b in zip(tup[0], tup[1]))
    ]
    # combo_iter = itertools.product(row_iter, col_iter)
    # for i in combo_iter:
    #     print(i)
    #     print(flatten(i))

    index_matrix = np.fromiter(
        map(flatten, itertools.product(row_iter, col_iter)),
        dtype=int,
        count=row_dim * col_dim,
    ).reshape((row_dim, col_dim))
    return index_matrix


# Expand naturally embedded Veronese elements into the Segre embedding for
# comparison with elements of a Segre variety
def partial_segre_expansion(dims, mults, expansion_scheme):
    # if np.all(np.array(mults) == 1):
    #     return np.arange(np.prod(dims))
    ex_arr = np.array(expansion_scheme, dtype=int)
    ex_sum = np.sum(ex_arr, axis=1)
    if not np.all(np.array(mults) == ex_sum):
        raise Exception(
            "Extraction must be a partition of multiplicities but {} != {}".format(
                ex_sum, mults
            )
        )
    num_extract = ex_arr.shape[1]
    lookup = {}
    for index, code in enumerate(ig.basis_generator(dims, mults)):
        lookup[code] = index
    extraction_gens = [
        ig.basis_generator(dims, ex_arr[:, i]) for i in range(num_extract)
    ]
    total = np.prod([ig.basis_size_for(dims, ex_arr[:, i]) for i in range(num_extract)])
    flatten = lambda prod_ind: lookup[
        tuple(
            tuple(sorted(ele for tup in tup_o_tups for ele in tup))
            for tup_o_tups in zip(*prod_ind)
        )
    ]
    indicies = np.fromiter(
        map(flatten, itertools.product(*extraction_gens)),
        dtype=int,
        count=total,
    )
    return indicies


# Greedy matching vectors by finding cosine similarity between pairs
def greedy_match_directions(dirs1, dirs2):
    nvecs1 = dirs1 / np.linalg.norm(dirs1, axis=0, keepdims=True)
    nvecs2 = dirs2 / np.linalg.norm(dirs2, axis=0, keepdims=True)
    dot_array = np.sum(nvecs1[:, :, None] * np.conj(nvecs2)[:, None, :], axis=0)
    # Clip values since rounding error occasionally produces value out of
    # bounds for large vectors
    dot_array = np.clip(dot_array, a_min=-1, a_max=1)
    work_table = np.abs(dot_array)
    num_pairs = min(dot_array.shape)
    permutations = np.zeros((2, num_pairs), dtype=int)
    for i in range(num_pairs):
        match_pt = np.unravel_index(np.argmax(work_table), dot_array.shape)
        permutations[0, i] = match_pt[0]
        permutations[1, i] = match_pt[1]
        work_table[match_pt[0], :] = 0
        work_table[:, match_pt[1]] = 0
    return permutations, dot_array


# Get array of absolute cosine similarities for the greedy matching
def similarity_between(directions1, directions2):
    permute, dot_array = greedy_match_directions(directions2, directions1)
    return np.abs(dot_array[permute[0], permute[1]])


# Generate a random subspace spanned by R unit vectors of which first S are
# flattened rank-1 tensors and columns (S+1) to R are generic vectors
def generate_XV_subspace(R, S, dims, mults, rng=rng):
    factors = [rng.normal(size=(d, S)) for d in dims]
    planted = khatri_rhao_products(factors, mults)
    basis = rng.normal(size=(planted.shape[0], R))
    basis[:, 0:S] = planted
    basis /= np.linalg.norm(basis, axis=0, keepdims=True)
    return basis, factors


# Computation of the M matrix when recovering solutions is not necessary
def get_unplanted_lift_segment(R, S, dims, mults, rng=rng):
    subspace, _ = generate_XV_subspace(R, S, dims, mults, rng=rng)
    lift = symmetric_lift(subspace, 2)
    if S > 0:
        plant_ind = np.arange(start=(R + 1), stop=(R - S + 1), step=-1)
        plant_ind[0] = 0
        lift_ind = np.cumsum(plant_ind)
        return np.delete(lift, lift_ind, axis=1)
    else:
        return lift


# TODO: rework this section to do a partial lift so it is easier to compute the
# M matrix


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    basis = rng.normal(size=(5, 4))
    symmetric_lift(basis, 2)
    pass
