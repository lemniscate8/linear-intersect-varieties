import unittest
import numpy as np
import linear_conic_intersect as lci
import subspace_manipulation as suma


class TestIntersect(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.tensor_signatures = [
            ([4, 3, 2], [1, 1, 1], 3),
            ([3], [3], 2),
            ([5, 3], [1, 2], 3),
            ([4, 3], [2, 1], 4),
        ]
        self.default_expansions = [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 1, 1]],
            [[1, 0, 0], [0, 1, 1]],
            [[1, 1, 0], [0, 0, 1]],
        ]

    def test_default_expansion(self):
        for sig, exp in zip(self.tensor_signatures, self.default_expansions):
            dims, mults, _ = sig
            default_exp = lci.default_expansion_method(dims, mults).tolist()
            # exp_arr = np.array(exp, dtype=int)
            self.assertEqual(
                exp,
                default_exp,
                msg="Default expansions not as expected.",
            )

    def test_partial_decomp_error(self):
        for dims, mults, rank in self.tensor_signatures:
            expand = lci.default_expansion_method(dims, mults)
            ncols = expand.shape[1]
            factors = [self.rng.normal(size=(d, rank)) for d in dims]
            flat_tensor = np.sum(suma.khatri_rhao_products(factors, mults), axis=1)
            pfactors = [
                suma.khatri_rhao_products(factors, expand[:, c]) for c in range(ncols)
            ]
            self.assertAlmostEqual(
                lci.partial_decomp_error(pfactors, flat_tensor, dims, mults, expand),
                0,
                msg="Partial decomposition incorrectly compared to flat tensor.",
            )

    def assertDecompDiagnosticsCheckOut(self, diagnostics):
        err, resids, match_dists = diagnostics
        self.assertAlmostEqual(
            max(match_dists),
            0,
            msg="Worst-case matching distance for two-factor matching is significantly above zero.",
        )
        self.assertAlmostEqual(
            max(resids),
            0,
            msg="Residuals in final factor computation significantly above zero.",
        )
        self.assertAlmostEqual(
            err,
            0,
            msg="Decomposition significantly deviates from given tensor in Frobenius norm.",
        )

    def test_jennrichs_algorithm_real(self):
        for dims, mults, rank in self.tensor_signatures:
            factors = [self.rng.normal(size=(d, rank)) for d in dims]
            flat_tensor = np.sum(suma.khatri_rhao_products(factors, mults), axis=1)
            _, _, diagnostics = lci.jennrich_partial_decomp(
                flat_tensor, dims, mults, rng=self.rng
            )
            self.assertDecompDiagnosticsCheckOut(diagnostics)

    def test_jennrichs_algorithm_complex(self):
        for dims, mults, rank in self.tensor_signatures:
            factors = [
                0.5 * self.rng.normal(size=(d, rank))
                + 0.5j * self.rng.normal(size=(d, rank))
                for d in dims
            ]
            flat_tensor = np.sum(suma.khatri_rhao_products(factors, mults), axis=1)
            _, _, diagnostics = lci.jennrich_partial_decomp(
                flat_tensor, dims, mults, rng=self.rng
            )
            self.assertDecompDiagnosticsCheckOut(diagnostics)


if __name__ == "__main__":
    unittest.main()
