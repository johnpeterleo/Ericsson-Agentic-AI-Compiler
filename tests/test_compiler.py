"""Compilation, diagnostic representations, and error propagation."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np

from src.compiler.compile import CompilationResult, benchmark, compile_program, simple_program


class CompilerIntegrationTests(unittest.TestCase):
    def test_compiled_program_matches_expected_values_and_exposes_ir(self):
        with jax.default_device(jax.devices("cpu")[0]):
            x = jnp.array([-1.0, 0.0, 2.0], dtype=jnp.float32)
            y = jnp.array([0.5, -2.0, 1.0], dtype=jnp.float32)
            compilation = compile_program(simple_program, (x, y))
            actual = compilation.executable(x, y).block_until_ready()

        expected = np.sin(np.asarray(x)) + np.asarray(y) * 2
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(actual.shape, x.shape)
        self.assertEqual(actual.dtype, x.dtype)
        self.assertIsInstance(compilation.stablehlo, str)
        self.assertTrue(compilation.stablehlo.strip())
        self.assertIsInstance(compilation.optimized_hlo, str)
        self.assertTrue(compilation.optimized_hlo.strip())


class CompilerContractTests(unittest.TestCase):
    def setUp(self):
        self.program = Mock()
        self.inputs = (np.array([1.0], dtype=np.float32),)
        self.executable = Mock()
        self.executable.as_text.return_value = "optimized representation"
        self.lowered = Mock()
        self.lowered.as_text.return_value = "lowered representation"
        self.lowered.compile.return_value = self.executable
        self.jit_patch = patch("src.compiler.compile.jax.jit")
        self.jit = self.jit_patch.start()
        self.addCleanup(self.jit_patch.stop)
        self.jit.return_value.lower.return_value = self.lowered

    def test_lowers_and_compiles_once_without_invoking_executable(self):
        result = compile_program(self.program, self.inputs)

        self.jit.assert_called_once_with(self.program)
        self.jit.return_value.lower.assert_called_once_with(*self.inputs)
        self.lowered.compile.assert_called_once_with()
        self.executable.assert_not_called()
        self.assertIs(result.executable, self.executable)
        self.assertEqual(result.stablehlo, "lowered representation")
        self.assertEqual(result.optimized_hlo, "optimized representation")

    def test_unavailable_optimized_hlo_is_preserved(self):
        self.executable.as_text.return_value = None
        result = compile_program(self.program, self.inputs)
        self.assertIsNone(result.optimized_hlo)
        self.assertIs(result.executable, self.executable)

    def test_lowering_error_propagates_unchanged(self):
        error = ValueError("lowering failed")
        self.jit.return_value.lower.side_effect = error
        with self.assertRaises(ValueError) as raised:
            compile_program(self.program, self.inputs)
        self.assertIs(raised.exception, error)
        self.lowered.compile.assert_not_called()

    def test_backend_compilation_error_propagates_unchanged(self):
        error = RuntimeError("backend compilation failed")
        self.lowered.compile.side_effect = error
        with self.assertRaises(RuntimeError) as raised:
            compile_program(self.program, self.inputs)
        self.assertIs(raised.exception, error)

    def test_inspection_errors_propagate_unchanged(self):
        for inspect in (self.lowered.as_text, self.executable.as_text):
            with self.subTest(inspect=inspect):
                error = RuntimeError("inspection failed")
                inspect.side_effect = error
                with self.assertRaises(RuntimeError) as raised:
                    compile_program(self.program, self.inputs)
                self.assertIs(raised.exception, error)
                inspect.side_effect = None


class CompilerDemoTests(unittest.TestCase):
    def test_demo_reports_unavailable_optimized_hlo(self):
        executable = Mock()
        executable.return_value.shape = (1000, 1000)
        compilation = CompilationResult(executable, stablehlo="lowered text", optimized_hlo=None)
        output = io.StringIO()
        with patch("src.compiler.compile.compile_program", return_value=compilation) as compiler:
            with redirect_stdout(output):
                benchmark()
        compiler.assert_called_once()
        self.assertEqual(executable.call_count, 2)
        self.assertIn("=== StableHLO ===", output.getvalue())
        self.assertIn("lowered text", output.getvalue())
        self.assertIn("Optimized HLO is unavailable for this backend.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
