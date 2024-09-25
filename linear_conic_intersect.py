import numpy as np
import scipy as sp
from scipy.special import binom
from scipy import linalg as lg
import segre_veronese_ideal_generation as ig
import subspace_manipulation as suma
import itertools
import warnings

rng = np.random.default_rng()


# def normalized_factors(factors):
#     weights = np.ones(factors[0].shape[1])
#     normalized = []
#     for F in factors:
#         norms = np.linalg.norm(F, axis=0)
#         normalized.append(F / norms[None, :])
#         weights *= norms
#     order = np.argsort(-weights)
#     return weights, [F[:, order] for F in normalized]


# Return a tri-partition of multiplicities used to map a higher-order tensor
# to a third-order tensor
# Default method uses one multiplicity of last mode for 3rd mode, splits even
# multiplicities in half, and uses a simple dithering scheme to allocate odd
# multiplicities
def default_expansion_method(dims, mults):
    extract = np.zeros((len(dims), 3), dtype=int)
    mult_arr = np.array(mults, dtype=int)
    extract[-1, -1] = 1
    mult_arr -= extract[:, -1]
    odd_modes = np.nonzero(mult_arr % 2)[0]
    dither = np.zeros_like(mults)
    dither[odd_modes] = np.arange(odd_modes.size) % 2
    extract[:, 1] = (mult_arr + dither) // 2
    extract[:, 0] = mult_arr - extract[:, 1]
    return extract


def max_rank_for_jennrich(dims, expand):
    d1 = ig.basis_size_for(dims, expand[:, 0])
    d2 = ig.basis_size_for(dims, expand[:, 1])
    return min(d1, d2)


# Computes a partial decomposition of any flattened higher-order tensor with
# partial symmetries by expanding to a 3rd order tensor and running Jennrich's
# algorithm
# Warning: if symmetric modes are split in the expansion, the decomposition
# cannot not enforce these symmetries; determining how these factors should be
# remediated if not identical is left to the user, or future methods
def jennrich_partial_decomp(
    flat_tensor,
    dims,
    mults,
    expansion_scheme=None,
    expansion_method=default_expansion_method,
    atol=1e-9,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()
    mults_arr = np.array(mults)
    if expansion_scheme is None:
        expansion_scheme = expansion_method(dims, mults)
    expand_arr = np.array(expansion_scheme)
    if expand_arr.shape[1] != 3 or not np.all(mults_arr == np.sum(expand_arr, axis=1)):
        raise Exception("Tensor extraction must be a tri-partition of tensor modes.")

    # First matrization
    fp1 = suma.tensor_flattening_pattern(dims, mults, expand_arr[:, 2])
    matrized_tensor = flat_tensor[fp1]
    remain_mults = mults_arr - expand_arr[:, 2]

    # Two random slices
    slab_count = matrized_tensor.shape[1]
    weights = rng.normal(size=(slab_count, 2))
    if flat_tensor.dtype == complex:
        weights = 0.5 * weights.astype(complex)
        weights += 0.5j * rng.normal(size=(slab_count, 2))
    slices = matrized_tensor @ rng.normal(size=(slab_count, 2))
    # Matrize slices to diagonalize
    fp2 = suma.tensor_flattening_pattern(dims, remain_mults, expand_arr[:, 1])
    factor1, factor2, dists = simultaneous_diagonalize(
        slices[fp2, 0], slices[fp2, 1], atol=atol
    )
    # Use least squares to estimate remaining factors
    factor3, res = estimate_remaining_factor(
        (factor1, factor2), matrized_tensor, dims, remain_mults, expand_arr[:, 0:2]
    )

    factors = (factor1, factor2, factor3)
    error = partial_decomp_error(
        factors,
        flat_tensor,
        dims,
        mults,
        expand_arr,
    )
    diagnostics = (error, res, dists)
    # Return factors, the expansion used, and diagnostics
    return factors, diagnostics, expand_arr


def simultaneous_diagonalize(arr1, arr2, atol=1e-9):
    m1, *_ = lg.lstsq(arr1.T, arr2.T)
    m1eig, m1vecs = lg.eig(m1.T)
    m1_estimated_rank = np.sum(np.abs(m1eig) > atol)
    m2, *_ = lg.lstsq(arr2, arr1)
    m2eig, m2vecs = lg.eig(m2.conj().T)
    m2_estimated_rank = np.sum(np.abs(m2eig) > atol)
    if m1_estimated_rank != m2_estimated_rank:
        raise Exception(
            """Ill conditioned contractions supplied or tensor is overcomplete;
unable to accurately estimate or match components,
some eigenvalues too close to zero."""
        )
    perm, match_dists = greedy_match_eigenvalues(
        m1eig, m2eig, num_matches=m1_estimated_rank
    )
    return m1vecs[:, perm[0]], m2vecs.conj()[:, perm[1]], match_dists


# Match eigenvalues from Jennrich's algorithm by computing all pairwise product
# and finding rows and columns of entries closest to unity
def greedy_match_eigenvalues(eigs1, eigs2, num_matches=None):
    # TODO: don't know if this metric is optimal given the ways noise will be
    # multiplicative if it occurs
    dist_to_unity = np.abs(eigs1[:, None] * eigs2.conj()[None, :] - 1)
    num_pairs = min(dist_to_unity.shape) if num_matches is None else num_matches
    permutation = np.zeros((2, num_pairs), dtype=int)
    distances = np.zeros(num_pairs)
    for i in range(num_pairs):
        match_pt = np.unravel_index(np.argmin(dist_to_unity), dist_to_unity.shape)
        permutation[0, i] = match_pt[0]
        permutation[1, i] = match_pt[1]
        distances[i] = dist_to_unity[match_pt]
        dist_to_unity[match_pt[0], :] = np.infty
        dist_to_unity[:, match_pt[1]] = np.infty
    return permutation, distances


# Use a khatri-rhao product of first two factors of the partial decomposition
# to estimate the final factor
def estimate_remaining_factor(factors, matrized_tensor, dims, mults, expansion_scheme):
    segre_expansion = suma.partial_segre_expansion(dims, mults, expansion_scheme)
    expanded = matrized_tensor[segre_expansion, :]
    khatri_rhao = suma.khatri_rhao_products(factors)
    last_factors, res, _, _ = lg.lstsq(khatri_rhao, expanded)
    return last_factors.transpose(), res


# Expands the flat tensor to have redundancies matching the partial
# decomposition then computes the L2 norm between
# Note: this "overweights" some basis elements effectively since entries would
# be identified in the full decomposition
def partial_decomp_error(factors, flat_tensor, dims, mults, expansion):
    expansion = suma.partial_segre_expansion(dims, mults, expansion)
    expanded_flat = flat_tensor[expansion]
    components = suma.khatri_rhao_products(factors)
    estimated = np.sum(components, axis=1)
    return np.linalg.norm(expanded_flat - estimated)


# Method for determining the kernel that simply computes the dense matrix
# product and uses scipy's svd-based null space method
def extract_kernel_naive(sparse_projector, dense_basis, expected_solutions):
    dense_combos = sparse_projector @ dense_basis
    return lg.null_space(dense_combos), True


def jennrich_total_decomp(
    flat_tensor,
    dims,
    mults,
    expansion_scheme=None,
    expansion_method=default_expansion_method,
    atol=1e-9,
    rng=None,
):
    factors, diagnostics, expand = jennrich_partial_decomp(
        flat_tensor,
        dims,
        mults,
        expansion_scheme=expansion_scheme,
        expansion_method=expansion_method,
        atol=atol,
        rng=rng,
    )
    # TODO: do full decomposition


def decompose_pure_tensor(flat_tensor, dims, mults):
    # TODO: recursive decomposition for a pure tensor
    pass


# Recovers planted solutions in a basis by finding all linear combinations
# that result in partially symmetric rank-1 tensors of specified dimension
def demix_subspace(
    basis,
    dims,
    mults,
    recurse=0,
    kernel_method=extract_kernel_naive,
    expected_solutions=None,
    rng=None,
):
    embedding_dim, subspace_dim = basis.shape
    expected_embed = ig.basis_size_for(dims, mults)
    sv_string = ig.print_sv(dims, mults)
    if embedding_dim != expected_embed:
        raise Exception(
            "Embedding dimension of {} is not the correct size for planted elements of X={}".format(
                embedding_dim, sv_string
            )
        )
    max_subspace_dim = int(ig.max_subspace_demixable(dims, mults))
    if subspace_dim > max_subspace_dim:
        warnings.warn(
            """The JLV algorithm used can only safely guarantee solutions with a subspace of dimension at most {} for planted solutions lying in {}.""".format(
                max_subspace_dim, sv_string
            )
        )

    lifted_basis = suma.symmetric_lift(basis, 2)
    soln_cutout_matrix = ig.sv_poly_matrix(dims, mults)
    kernel_basis, found_all = kernel_method(
        soln_cutout_matrix, lifted_basis, expected_solutions
    )
    # Must normalize before reshaping and decomposing the tensor
    weights = suma.symmetric_weights(subspace_dim, 2)
    kernel_basis /= weights[:, None]
    kernel_dim = kernel_basis.shape[1]
    print("Kernel dim is ", kernel_dim)
    if (recurse > 0) and (kernel_dim > subspace_dim):
        coefs, diagnostics = demix_subspace(
            kernel_basis, [subspace_dim], [2], recurse=recurse - 1, rng=rng
        )
        return kernel_basis @ coefs, diagnostics
    flat_ktensor = kernel_basis.flatten()
    ktensor_dims = [subspace_dim, kernel_dim]
    ktensor_mults = [2, 1]

    mode_expand = [[1, 1, 0], [0, 0, 1]]

    factors, diagnostics, *_ = jennrich_partial_decomp(
        flat_ktensor,
        ktensor_dims,
        ktensor_mults,
        expansion_scheme=mode_expand,
        rng=rng,
    )
    # TODO: first two factors will be the same due to symmetry but should check
    # that and maybe symmetrize in case of numerical or other errors

    # print(factors[0] / factors[1])
    # First factor contains coefficient combinations that will result in rank-1
    # tensors
    return factors[0], diagnostics


def generateXVsubspace(R, S, dims, mults, rng=rng):
    factors = [rng.normal(size=(d, S)) for d in dims]
    planted = suma.khatri_rhao_products(factors, mults)
    basis = rng.normal(size=(planted.shape[0], R))
    basis[:, 0:S] = planted
    basis /= np.linalg.norm(basis, axis=0, keepdims=True)
    return basis, factors


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # # Standard segre product
    # dims = [4, 3, 5]
    # mults = [1, 1, 1]
    # rank = 6 # Overcomplete, should fail
    # factors = [rng.normal(size=(d, rank)) for d in dims]
    # nfacts = [factor / np.linalg.norm(factor, axis=0) for factor in factors]
    # for fact in nfacts:
    #     print(fact)
    # tensor_prods = suma.khatri_rhao_products(nfacts, mults)
    # # flattening = suma.tensor_flattening_pattern([4, 3, 2], [1, 1, 1], [0, 0, 1])
    # # one_tensor = tensor_prods[flattening, 0]
    # # print(one_tensor[:, 0] / one_tensor[:, 1])
    # flat_tensor = np.sum(tensor_prods, axis=1)
    # extraction = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # (
    #     factors,
    #     diagnostics,
    #     *_,
    # ) = jennrich_partial_decomp(flat_tensor, dims, mults, extraction, rng=rng)
    # print([factor.shape for factor in factors])
    # print(diagnostics)

    # # Veronese-Segre product
    # dims = [4, 4]
    # mults = [2, 1]
    # rank = 5
    # factors = [
    #     0.5 * rng.normal(size=(d, rank)) + 0.5j * rng.normal(size=(d, rank))
    #     for d in dims
    # ]
    # nfacts = [factor / np.linalg.norm(factor, axis=0) for factor in factors]
    # for fact in nfacts:
    #     print(fact)
    # tensor_prods = suma.khatri_rhao_products(nfacts, mults)

    # # print(one_tensor[:, 0] / one_tensor[:, 1])
    # flat_tensor = np.sum(tensor_prods, axis=1)
    # extraction = [[1, 1, 0], [0, 0, 1]]
    # (
    #     factors,
    #     diagnostics,
    #     *_,
    # ) = jennrich_partial_decomp(flat_tensor, dims, mults, extraction, rng=rng)
    # print([factor.shape for factor in factors])
    # print(diagnostics)
    # print(suma.similarity_between(nfacts[0], factors[0]))
    # print(suma.similarity_between(nfacts[0], factors[1]))
    # print(suma.similarity_between(nfacts[1], factors[2]))

    # # Test flattening of tensors
    # dims = [4, 5]
    # mults = [1, 1]
    # rank = 3
    # factors = [rng.normal(size=(d, rank)) for d in dims]
    # tensor = suma.khatri_rhao_products(factors)
    # tdims = dims + [rank]
    # print(tdims)
    # tmults = mults + [1]
    # factors, diagnostics, expansion = jennrich_partial_decomp(
    #     tensor.flatten(), tdims, tmults
    # )
    # print(expansion)
    # print(diagnostics)
    # print([factor.shape for factor in factors])

    # # Test subspace demixing
    # dims = [5, 5]
    # mults = [1, 1]
    # R = ig.max_subspace_demixable(dims, mults)
    # # R = 4
    # plant_ratio = 1
    # S = int(R * plant_ratio)
    # # S = 3
    # print(R, S)
    # raw_basis, factors = generateXVsubspace(R, S, dims, mults)
    # column_mixer = rng.normal(size=(R, R))
    # basis = raw_basis @ column_mixer
    # coefs, diagnostics = demix_subspace(basis, dims, mults, rng=rng)
    # print("Coef shape:", coefs.shape)
    # print(diagnostics)
    # recovered_plants = basis @ coefs

    # print(column_mixer @ coefs)
    # matching = suma.similarity_between(raw_basis[:, :S], recovered_plants)
    # print(matching)

    # for val in itertools.combinations_with_replacement(range(3), 3):
    #     print(val)
    index_iter = itertools.combinations_with_replacement(range(4), 2)
    # vals = tuple(zip(*index_iter)
    # print(np.arange(27).reshape((3, 3, 3))[vals])
    for val in index_iter:
        print(val)
        print(val[1] + ((val[0] * (val[0] + 1)) // 2))
