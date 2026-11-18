"""Teaching content must remain visible when the slide layout changes."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .build import flow_labels, insert_code_slides, result_rows, write_html
from .content import Slide, create_slides, ensure_pairs
from .test_presentation import fixture


class VisualLayoutTests(unittest.TestCase):
    def test_every_method_code_example_is_visible_and_keeps_its_narration(self):
        root = Path(__file__).resolve().parents[1]
        sources = [
            "src/fraudlab/bpe.py", "src/fraudlab/frequency.py", "src/fraudlab/offset.py",
            "src/fraudlab/pooled.py", "src/fraudlab/joints.py", "src/fraudlab/neural_offset.py",
            "src/fraudlab/strings.py", "scripts/train_nested_stack.py", "src/fraudlab/realmlp.py",
        ]
        specifications = [(relative, (root / relative).read_text(encoding="utf-8").splitlines()[0],
                           1, "Code fixture", [], "Code fixture narration") for relative in sources]
        slides = [Slide("test", "Submissão e posição observadas", [], "test")]
        with patch("presentation.build.method_code_specs", return_value=specifications):
            insert_code_slides(slides, root, required=True)
        ensure_pairs(slides)
        code_examples = [slides[i + 1] for i, slide in enumerate(slides[:-1])
                         if slide.code_reference.split(":")[0] in sources]
        self.assertEqual(len(code_examples), 9)
        for slide in code_examples:
            with self.subTest(title=slide.title):
                self.assertEqual(slide.kind, "analogy")
                self.assertEqual(slide.points, [slide.narration])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_html(slides, output, draft=True)
            document = (output / "slides.html").read_text(encoding="utf-8")
            self.assertEqual(document.count('class="analogy-layout"'), 11)
            for slide in code_examples:
                self.assertIn("<li>" + slide.narration + "</li>", document)

    def test_architecture_diagram_uses_the_executed_dimensions(self):
        evidence, run = fixture()
        slides = create_slides(evidence, [run])
        architecture = next(slide for slide in slides if slide.visual == "architecture")
        self.assertEqual(flow_labels(architecture), ["3 entradas", "8 unidades", "4 unidades", "1 saída · sigmoid"])

    def test_result_table_preserves_full_names_and_recorded_metric_strings(self):
        name = "round-fixture-long-name<&>"
        slide = Slide("test", "Resultados: 5 folds, 100 OOF", [name + ": AUC OOF 0,95170"], "fixture")
        self.assertEqual(result_rows(slide), [(name, "0,95170")])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_html([slide], output, draft=True)
            document = (output / "slides.html").read_text(encoding="utf-8")
            self.assertIn("<td>round-fixture-long-name&lt;&amp;&gt;</td><td>0,95170</td>", document)

    def test_older_slide_json_can_still_be_opened(self):
        old = {"chapter": "test", "title": "test", "points": [], "narration": "test", "visual": "architecture"}
        self.assertEqual(flow_labels(Slide(**old)), ["Entradas", "Camadas ocultas", "Sigmoid", "Probabilidade"])


if __name__ == "__main__":
    unittest.main()
