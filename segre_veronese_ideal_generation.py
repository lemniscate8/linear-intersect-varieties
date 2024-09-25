import itertools
from queue import PriorityQueue
import numpy as np
from scipy.special import binom
from scipy import sparse


# Generates a linearly independent set of polynomials that cut out a
# segre-veronese variety
def sv_poly_generator(dims, mults):
    for root_code in forest_roots(dims, mults):
        for edge in spanning_tree_for(root_code):
            yield tuple(tuple(zip(*node)) for node in edge)


# Root codes determine a partition of the ideal of polynomials simplifying
# generation of polynomials
def forest_roots(dims, mults):
    return itertools.product(
        *[
            itertools.combinations_with_replacement(range(d), 2 * m)
            for d, m in zip(dims, mults)
        ]
    )


# The polynomials that cut out a segre-veronese variety can be partitioned
# such that a spanning tree grown on each partition ensures independence
def spanning_tree_for(root_code):
    root = tuple(
        (group[: (len(group) // 2)], group[(len(group) // 2) :]) for group in root_code
    )
    # print("Root is", root)
    connected = set()
    seach_next = PriorityQueue()
    seach_next.put(root)
    while not seach_next.empty():
        active = seach_next.get()
        connected.add(active)
        for neighbor in neighbors_of(active):
            if neighbor in connected:
                continue
            seach_next.put(neighbor)
            connected.add(neighbor)
            yield (active, neighbor)


# Iterator to determine all possible neighbors of a degree 2 monomial such
# that the difference is a polynomial in the prime ideal of a segre-veronese
# variety
def neighbors_of(node):
    num_compartments = len(node)
    for group in range(num_compartments):
        pre = node[:group]
        post = node[(group + 1) :]
        for alt_group in multiset_swaps_between(node[group]):
            if alt_group == node[group]:
                continue
            new_node = pre + (alt_group,) + post
            canonical_adjoint = sorted(zip(*new_node))
            canonical_new = tuple(zip(*canonical_adjoint))
            if canonical_new == node:
                continue
            yield canonical_new


# Iterator to determine all the ways of exchanging elements between sorted
# multisets
def multiset_swaps_between(mset_pair):
    mset_left, mset_right = mset_pair
    # print("MsetL:", mset_left)
    # print("MsetR:", mset_right)
    u_left = np.unique(mset_left)
    u_right = np.unique(mset_right)
    for val_left in u_left:
        for val_right in u_right:
            if val_left == val_right:
                continue
            left_new = exchange(mset_left, val_left, val_right)
            right_new = exchange(mset_right, val_right, val_left)
            yield left_new, right_new


# Helper function to swap elements out of a sorted multiset
def exchange(mset, val_out, val_in):
    mset_arr = np.array(mset)
    side = "left" if val_out < val_in else "right"
    loc_out = np.searchsorted(mset_arr, val_out, side=side)
    loc_in = np.searchsorted(mset_arr, val_in, side=side)
    if val_out < val_in:
        mset_arr[loc_out : (loc_in - 1)] = mset_arr[(loc_out + 1) : loc_in]
        mset_arr[loc_in - 1] = val_in
    else:
        mset_arr[(loc_in + 1) : loc_out] = mset_arr[loc_in : (loc_out - 1)]
        mset_arr[loc_in] = val_in
    return tuple(mset_arr)


# Dimension of embedding space
def basis_size_for(dims, mults):
    multarr = np.array(mults, dtype=int)
    dimarr = np.array(dims, dtype=int)
    return int(np.prod(binom(dimarr + multarr - 1, multarr)))


# The maximum dimension of a subspace for which the JLV
# algorithm will successfully find all planted solutions
# given the dimensions and multiplicities of a Segre-Veronese
# variety
def max_subspace_demixable(dims, mults=None):
    dimarr = np.array(dims, dtype=int)
    multarr = np.array(mults, dtype=int) if mults is not None else 1
    base_dims = dimarr + multarr - 1
    sym_dims = dimarr + 2 * multarr - 1
    deficits = binom(sym_dims, 2 * multarr) / binom(base_dims, multarr)
    bigN = np.prod(binom(base_dims, multarr))
    return int((bigN + 1) / 2 - np.prod(deficits))


def sv_independent_poly_count(dims, mults):
    multarr = np.array(mults, dtype=int)
    dimarr = np.array(dims, dtype=int)
    N = basis_size_for(dims, mults)
    factor_redund = np.prod(binom(dimarr + 2 * multarr - 1, 2 * multarr))
    return int(binom(N + 1, 2) - factor_redund)


# Produces generates indicies for all basis vectors (this doesn't make sense)
def basis_generator(dims, mults):
    return itertools.product(
        *[
            itertools.combinations_with_replacement(range(d), m)
            for d, m in zip(dims, mults)
        ]
    )


# An ordering for the degree 2 monomials needed to cut out a segre-veronese
# variety
def sv_monomial_dictionary(dims, mults):
    base_tuple_gen = itertools.product(
        *[
            itertools.combinations_with_replacement(range(d), m)
            for d, m in zip(dims, mults)
        ]
    )
    lut = {}
    for i, tuple in enumerate(
        itertools.combinations_with_replacement(base_tuple_gen, 2)
    ):
        lut[tuple] = i
    return lut


# A matrix that can be applied to a 2nd symmetric lift of subspace to evaluate
# the polynomials that cut out a segre-veronese variety against a linear
# subspace
# Use is that ker(sv_poly_matrix * symmetric_lift(U,2)) is a subspace of rank-1
# elements in U
def sv_poly_matrix(dims, mults):
    num_rows = sv_independent_poly_count(dims, mults)
    N = basis_size_for(dims, mults)
    num_cols = int(binom(N + 1, 2))
    monomial_lookup = sv_monomial_dictionary(dims, mults)
    poly_mat = sparse.lil_array((num_rows, num_cols), dtype=int)
    for i, (node1, node2) in enumerate(sv_poly_generator(dims, mults)):
        # print(node1, node2)
        poly_mat[i, monomial_lookup[node1]] = 1
        poly_mat[i, monomial_lookup[node2]] = -1
    return poly_mat


def print_sv(dims, mults=None, latex=False):
    if latex:
        if mults is None or np.all(np.array(mults) == 1):
            return (
                "\operatorname{SV}(\mathbb{P}^{"
                + "},\mathbb{P}^{".join([str(d - 1) for d in dims])
                + "})"
            )
        return (
            "SV_{"
            + ",".join([str(m) for m in mults])
            + "}(\mathbb{P}^{"
            + "},\mathbb{P}^{".join([str(d - 1) for d in dims])
            + "})"
        )
    if mults is None or np.all(np.array(mults) == 1):
        return "SV(" + ", ".join([str(d) for d in dims]) + ")"
    return (
        "SV("
        + ",".join(
            [str(d) + ("" if m == 1 else "^" + str(m)) for d, m in zip(dims, mults)]
        )
        + ")"
    )


if __name__ == "__main__":

    # for i, edge in enumerate(poly_generator([2], [3])):
    #     print(i, ":", edge)
    lut = sv_monomial_dictionary([3], [3])
    for key in lut:
        print(key, lut[key])

    # polymat = sv_poly_matrix([2, 2, 2, 2], [1, 1, 1, 1])
    # monomial_degrees = np.ones(polymat.shape[0]) @ np.abs(polymat)
    # print(monomial_degrees)
    # np.savetxt("view.csv", polymat.toarray(), delimiter=" ", fmt="%d")
