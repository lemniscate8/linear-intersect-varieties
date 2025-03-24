import numpy as np
import scipy as sp
from scipy.special import binom
from scipy import linalg as lg
from scipy import sparse
import segre_veronese_ideal_generation as ig
import subspace_manipulation as suma
import itertools
import warnings

rng = np.random.default_rng()


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
    # contraction_vectors=None,
    expansion_scheme=None,
    expansion_method=default_expansion_method,
    atol=1e-9,
    expected_rank=None,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()
    mults_arr = np.array(mults)
    if expansion_scheme is None:
        expansion_scheme = expansion_method(dims, mults)
        # print(expansion_scheme)
    expand_arr = np.array(expansion_scheme)
    if expand_arr.shape[1] != 3 or not np.all(mults_arr == np.sum(expand_arr, axis=1)):
        raise Exception(
            "Tensor expansion array must be a tri-partition of tensor modes."
        )

    # First matrization
    fp1 = suma.tensor_flattening_pattern(dims, mults, expand_arr[:, 2])

    matrized_tensor = flat_tensor[fp1]
    remain_mults = mults_arr - expand_arr[:, 2]

    # Two random slices
    slab_count = matrized_tensor.shape[1]
    fp2 = suma.tensor_flattening_pattern(dims, remain_mults, expand_arr[:, 1])

    dists = None
    if slab_count <= 1:
        # If tensor has degenerate shape, find best rank-1 approximation via SVD
        matrix = matrized_tensor[fp2, 0]
        u, s, vh = lg.svd(matrix)
        factor1 = s[0] * u[:, 0:1]
        factor2 = vh[0:1, :].transpose()
    else:
        weights = rng.normal(size=(slab_count, 2))
        if flat_tensor.dtype == complex:
            weights = 0.5 * weights.astype(complex)
            weights += 0.5j * rng.normal(size=(slab_count, 2))
        weights /= np.linalg.norm(weights, axis=1, keepdims=True)
        slices = matrized_tensor @ weights

        # Matrize slices to diagonalize

        factor1, factor2, dists = simultaneous_diagonalize(
            slices[fp2, 0],
            slices[fp2, 1],
            atol=atol,
            match_tol=atol,
            expected_rank=expected_rank,
        )
        # print("Factor 1 shape:", factor1.shape)
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


# Subroutine to perform simultaneous diagonalization for two matrices
def simultaneous_diagonalize(arr1, arr2, atol=1e-9, match_tol=1e-9, expected_rank=None):
    m1, *_ = lg.lstsq(arr1.T, arr2.T)
    m1eig, m1vecs = lg.eig(m1.T)
    m2, *_ = lg.lstsq(arr2, arr1)
    m2eig, m2vecs = lg.eig(m2.conj().T)

    # Re-evaluate: this might be more robustly detected by poor matchings
    # Investigate: ranks are larger than expected, is there numerical error
    # accumulating from the scipy functions I'm using?
    m1_estimated_rank = np.sum(np.abs(m1eig) > atol)
    m2_estimated_rank = np.sum(np.abs(m2eig) > atol)
    estimated_matches = min(m1_estimated_rank, m2_estimated_rank)
    #     if m1_estimated_rank != m2_estimated_rank:
    #         raise Exception(
    #             """Bad contraction or tensor is overcomplete;
    # unable to accurately estimate or match components,
    # some eigenvalues too close to zero."""
    #         )
    perm, match_dists = greedy_match_eigenvalues(
        m1eig,
        m2eig,
        num_matches=expected_rank if expected_rank is not None else estimated_matches,
    )
    if expected_rank is None:
        keepers = match_dists < match_tol
        if np.all(~keepers):
            raise np.linalg.LinAlgError(
                "No eigenvalues match above tolerance in simultaneous diagonalization.",
                match_dists,
            )
    else:
        keepers = np.s_[:]
    return (
        m1vecs[:, perm[0, keepers]],
        m2vecs.conj()[:, perm[1, keepers]],
        match_dists[keepers],
    )


# Match eigenvalues from Jennrich's algorithm robustly
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
        dist_to_unity[match_pt[0], :] = np.inf
        dist_to_unity[:, match_pt[1]] = np.inf
    return permutation, distances


# Use a khatri-rhao product of first two factors of the partial decomposition
# and linear least squares to estimate the final factor
def estimate_remaining_factor(factors, matrized_tensor, dims, mults, expansion_scheme):
    segre_expansion = suma.partial_segre_expansion(dims, mults, expansion_scheme)
    expanded = matrized_tensor[segre_expansion, :]
    khatri_rhao = suma.khatri_rhao_products(factors)
    last_factors, res, _, _ = lg.lstsq(khatri_rhao, expanded)
    return last_factors.transpose(), res


# Expand the flat tensor to have redundancies matching the partial
# decomposition then computes the L2 norm between
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
    if dense_combos.shape[1] == 1:
        if np.linalg.norm(dense_combos) < 1e-9:
            return np.array([[1.0]]), True
    return lg.null_space(dense_combos), True


# WIP method for decomposing higher order tensors
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
    pass


# An implementation of the JLV algorithm
# Recovers planted solutions in a basis by finding all linear combinations
# that result in partially symmetric rank-1 tensors of specified dimension
def demix_subspace(
    basis,
    dims,
    mults,
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
    # if subspace_dim > max_subspace_dim:
    #     warnings.warn(
    #         """The JLV algorithm used can only safely guarantee solutions with a subspace of dimension at most {} for planted solutions lying in {}.""".format(
    #             max_subspace_dim, sv_string
    #         )
    #     )
    lifted_basis = suma.symmetric_lift(basis, 2)
    soln_cutout_matrix = ig.sv_poly_matrix(dims, mults)
    kernel_basis, found_all = kernel_method(
        soln_cutout_matrix, lifted_basis, expected_solutions
    )
    # Must normalize before reshaping and decomposing the tensor
    weights = suma.symmetric_weights(subspace_dim, 2)
    kernel_basis /= weights[:, None]
    kernel_dim = kernel_basis.shape[1]
    if kernel_dim == 0:
        # Code 1 means no planted solutions detected, only trivial intersection
        return np.zeros((subspace_dim)), (1, None)
    flat_ktensor = kernel_basis.flatten()
    ktensor_dims = [subspace_dim, kernel_dim]
    ktensor_mults = [2, 1]
    mode_expand = [[1, 1, 0], [0, 0, 1]]
    factors, diagnostics, *_ = jennrich_partial_decomp(
        flat_ktensor,
        ktensor_dims,
        ktensor_mults,
        expansion_scheme=mode_expand,
        expected_rank=expected_solutions,
        rng=rng,
    )
    # TODO: first two factors will be the same due to symmetry but should check
    # that and maybe symmetrize in case of numerical or other errors

    # print(factors[0] / factors[1])
    # First factor contains coefficient combinations that will result in rank-1
    # tensors

    # Code 0 indicates some planted solutions were found and decomposed
    return factors[0], (0, diagnostics)
