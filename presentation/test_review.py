"""Check that mechanical QA rejects incorrect media timing."""

from pathlib import Path
import tempfile
import unittest

from presentation.build import sha256
from presentation.review import read_captions, validate_timeline, verify_artifacts


class ReviewTests(unittest.TestCase):
    def test_modified_timeline_is_rejected_before_consumption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            timeline = root / "timeline.json"
            timeline.write_text("[]", encoding="utf-8")
            recorded = {"timeline.json": sha256(timeline)}
            verify_artifacts(root, recorded)
            timeline.write_text('[{"slide":1}]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "timeline.json"):
                verify_artifacts(root, recorded)

    def test_caption_formats_represent_the_same_times(self):
        with tempfile.TemporaryDirectory() as directory:
            srt, vtt = (Path(directory) / name for name in ("captions.srt", "captions.vtt"))
            srt.write_text("1\n00:00:00,100 --> 00:00:01,000\nNarração.\n", encoding="utf-8")
            vtt.write_text("WEBVTT\n\n1\n00:00:00.100 --> 00:00:01.000\nNarração.\n", encoding="utf-8")
            self.assertEqual(read_captions(srt, 2), read_captions(vtt, 2))

    def test_caption_cannot_extend_beyond_video(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.srt"
            path.write_text("1\n00:00:00,000 --> 00:00:05,000\nNarração.\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside"):
                read_captions(path, 2)

    def test_numeric_spoken_content_is_not_mistaken_for_cue_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.srt"
            path.write_text("1\n00:00:00,000 --> 00:00:01,000\n128\n", encoding="utf-8")
            self.assertEqual(read_captions(path, 2)[0]["text"], "128")

    def test_timeline_must_cover_every_slide_without_gaps(self):
        valid = [{"slide": 1, "start": 0, "duration": 2}, {"slide": 2, "start": 2, "duration": 3}]
        validate_timeline(valid, 2, 5)
        with self.assertRaisesRegex(ValueError, "every slide"):
            validate_timeline(valid, 3, 5)
        valid[1]["start"] = 2.5
        with self.assertRaisesRegex(ValueError, "consecutive"):
            validate_timeline(valid, 2, 5)


if __name__ == "__main__":
    unittest.main()
