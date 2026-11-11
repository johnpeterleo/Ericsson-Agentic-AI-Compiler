"""The compiled-form summary counts what XLA launches and compares it with the reference."""

import unittest

import numpy as np

from src.agent.feedback import CompiledSummary, format_op_diff, summarize_compiled, summarize_hlo

HLO = """HloModule jit_example

%fused_computation (param_0: f32[4]) -> f32[4] {
  %param_0 = f32[4]{0} parameter(0)
  ROOT %exponential.1 = f32[4]{0} exponential(%param_0)
}

%body (arg: (s32[], f32[4])) -> (s32[], f32[4]) {
  %arg = (s32[], f32[4]{0}) parameter(0)
  %dot.2 = f32[4]{0} dot(%x, %y), metadata={op_name="jit(example)/dot" source_line=12}
  ROOT %tuple.1 = (s32[], f32[4]{0}) tuple(%i, %dot.2)
}

ENTRY %main.3 (x: f32[4]) -> bf16[4] {
  %x = f32[4]{0} parameter(0)
  %exp_fusion = f32[4]{0} fusion(%x), kind=kLoop, calls=%fused_computation
  %gemm = f32[4]{0} custom-call(%x), custom_call_target="__cublas$gemm", backend_config={"alpha":{"real":1}}
  %while.1 = (s32[], f32[4]{0}) while(%t), condition=%cond, body=%body
  ROOT %convert = bf16[4]{0} convert(%exp_fusion)
}
"""


def summary_with(op_counts):
    return CompiledSummary(op_counts, ("f32",), None, None, "")


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.summary = summarize_hlo(HLO, {"flops": 8.0})

    def test_counts_launched_operations_but_not_fused_bodies(self):
        self.assertEqual(self.summary.op_counts, {
            "fusion(kLoop)": 1, "custom-call(__cublas$gemm)": 1, "while": 1, "convert": 1, "dot": 1,
        })

    def test_finds_every_floating_point_type(self):
        self.assertEqual(self.summary.float_dtypes, ("bf16", "f32"))

    def test_excerpt_drops_metadata_and_backend_config(self):
        self.assertNotIn("metadata", self.summary.excerpt)
        self.assertNotIn("backend_config", self.summary.excerpt)
        self.assertIn('custom_call_target="__cublas$gemm"', self.summary.excerpt)

    def test_reads_the_cost_analysis(self):
        self.assertEqual((self.summary.flops, self.summary.bytes_accessed), (8.0, None))

    def test_diff_lists_only_changed_operations(self):
        reference = summary_with({"dot": 3, "while": 1, "fusion(kLoop)": 2})
        candidate = summary_with({"dot": 1, "fusion(kLoop)": 2, "reduce": 1})

        self.assertEqual(format_op_diff(reference, candidate), "dot 3 -> 1, while 1 -> 0, reduce 0 -> 1")

    def test_diff_of_equal_programs_says_so(self):
        summary = summary_with({"dot": 1})
        self.assertEqual(format_op_diff(summary, summary), "same operation counts as the reference")

    def test_summarizes_a_real_compiled_program(self):
        summary = summarize_compiled(lambda a, b: a @ b, (np.ones((4, 4), np.float32),) * 2)

        self.assertIn("dot", summary.op_counts)
        self.assertEqual(summary.float_dtypes, ("f32",))


if __name__ == "__main__":
    unittest.main()
