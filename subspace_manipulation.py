import itertools
import numpy as np
from scipy.special import binom, factorial
from scipy.linalg import khatri_rao
from scipy import sparse
import segre_veronese_ideal_generation as ig


def khatri_rhao_products(factors, mults=None):
    if (mults is None) or (np.all(np.array(mults, dtype=int) == 1)):
        tensors = np.ones((1, factors[0].shape[1]))
        for factor_mat in factors:
            tensors = khatri_rao(tensors, factor_mat)
        return tensors
    bases = [
        symmetric_khatri_rhao(factor, mult) for factor, mult in zip(factors, mults)
    ]
    return khatri_rhao_products(bases)


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


def symmetric_lift(basis, k):
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
            prod = basis[row_iter[:, index, None], col_iter[:, perm[index]]] * prod
        lifted_basis += prod
        k_fact += 1
    return lifted_basis / k_fact


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


# Produces a tall or square matrix that is a look up table balanced flattening
# of a symmetric tensor
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


# Greedy matching vectors by finding angles between all pairs
def greedy_match_directions(dirs1, dirs2):
    nvecs1 = dirs1 / np.linalg.norm(dirs1, axis=0, keepdims=True)
    nvecs2 = dirs2 / np.linalg.norm(dirs2, axis=0, keepdims=True)
    dot_array = np.sum(nvecs1[:, :, None] * np.conj(nvecs2)[:, None, :], axis=0)
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


# Get array of angles between lines after greedy matchings
def similarity_between(directions1, directions2):
    permute, dot_array = greedy_match_directions(directions2, directions1)
    return np.abs(dot_array[permute[0], permute[1]])


if __name__ == "__main__":
    # basis = np.array([[1, 2, 0], [0, 1, 1]])
    # print(basis)
    # basis_lift = symmetric_lift2(basis)
    # print(basis_lift)
    # arr = np.array(list(itertools.combinations_with_replacement(range(3), 2)))
    # print(arr)
    # print(list(itertools.combinations_with_replacement(range(3), 2)))
    # basis = np.array([[1, 2], [0, 1], [1, 0]])
    # sp_basis = sparse.csr_array(basis)
    # print(basis)

    # # Test symmetric lift
    # basis2 = symmetric_lift(basis, 2)
    # print(basis2)
    # sp_basis2 = symmetric_lift(sp_basis, 2)
    # print(sp_basis2.data)
    # print(sp_basis2.toarray())

    # # Test symmetric khatri-rhao
    # kr2 = symmetric_khatri_rhao(basis, 2)
    # print(kr2)
    # sp_kr2 = symmetric_khatri_rhao(sp_basis, 2)
    # print(sp_kr2.toarray())

    # # Test tensor product
    # tensors = tensor_products([basis, basis])
    # print(tensors)
    # sp_tensors = tensor_products([sp_basis, sp_basis])
    # print(type(sp_tensors))

    # dim = 3
    # mult = 3
    # rng = np.random.default_rng()
    # sym_tens_vals = rng.normal(size=int(binom(dim + mult - 1, mult)))
    # print(sym_tens_vals)
    # lut = tensor_flattening_pattern([dim], [mult], [1])
    # print(lut)
    # print(sym_tens_vals[lut])
    # lut1 = tensor_flattening_pattern([3], [3], [1])
    # print(lut1)
    # lut2 = tensor_flattening_pattern([4, 3], [1, 1], [0, 1])
    # print(lut2)
    # print(np.arange(12).reshape((4, 3)))
    # lut3 = tensor_flattening_pattern([3], [2], [1])
    # lut = tensor_flattening_pattern([3, 3], [2, 1], [0, 1])
    # factors = [rng.normal(size=(3, 1)) for i in range(2)]
    # print(factors)
    # tensor = tensor_products(factors, [2, 1])
    # # print(tensor)
    # flattened = tensor[lut, 0]
    # print(flattened[0, :] / flattened[1, :])
    # print(flattened[:, 0] / flattened[:, 1])
    partition = np.array([[1], [1], [1]])
    print(partition)
    ind = partial_segre_expansion([2], [3], [[2, 1]])
    # ind = partial_segre_expansion(
    #     [2, 2],
    #     [3, 1],
    #     [[1, 1, 1], [0, 0, 1]],
    # )
    print(ind)
