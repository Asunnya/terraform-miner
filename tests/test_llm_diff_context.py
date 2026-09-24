import pytest

from src.llm_analysis.diff_context import (
    ContextLimits,
    allocate_budget,
    render_diffs,
    split_hunks,
    truncate_patch,
)
from src.llm_analysis.schemas import FileDiff

BIG = ContextLimits(max_diff_chars=10_000, max_chars_per_file=10_000, max_files_per_commit=50,
                    max_listed_file_names=100)


def _hunk(n, lines=5):
    body = "".join(f"-old_{n}_{i}\n+new_{n}_{i}\n" for i in range(lines))
    return f"@@ -{n},{lines} +{n},{lines} @@ resource\n{body}"


def test_render_preserves_diff_semantics_and_file_names():
    patch = '@@ -1,3 +1,3 @@\n resource "aws_subnet" "a" {\n-  cidr_block = "10.0.0.0/16"\n+  cidr_block = "10.0.0.0/24"\n }\n'
    rendered = render_diffs([FileDiff("modules/network/main.tf", patch)], [], BIG)

    assert "FILE: modules/network/main.tf\n```diff\n" + patch + "```" in rendered.text
    assert not rendered.diff_truncated
    assert rendered.original_diff_chars == rendered.submitted_diff_chars == len(patch)


def test_render_is_deterministic():
    diffs = [FileDiff("a.tf", _hunk(1)), FileDiff("b.tf", _hunk(2), also_in=["c.tf"])]
    limits = ContextLimits(200, 150, 1, 100)
    assert render_diffs(diffs, ["d.tf"], limits) == render_diffs(diffs, ["d.tf"], limits)


def test_split_hunks():
    patch = _hunk(1) + _hunk(20)
    assert split_hunks(patch) == [_hunk(1), _hunk(20)]
    assert split_hunks("Binary files differ\n") == ["Binary files differ\n"]


def test_truncate_keeps_whole_hunks_then_cuts_at_line_boundary():
    patch = _hunk(1) + _hunk(20) + _hunk(40)
    limit = len(_hunk(1)) + 60
    kept, full, total = truncate_patch(patch, limit)

    assert kept.startswith(_hunk(1))
    assert len(kept) <= limit
    assert kept.endswith("\n")  # nenhuma linha cortada no meio
    assert kept[len(_hunk(1)):].startswith("@@ -20,5")
    assert (full, total) == (1, 3)


def test_truncate_single_huge_line_is_cut_mid_line():
    kept, full, _ = truncate_patch("@" * 1000, 100)
    assert kept == "@" * 100 and full == 0


def test_allocate_budget_water_filling():
    # arquivos pequenos recebem tudo; o resto divide o que sobra
    assert allocate_budget([100, 5000, 5000], total=1100, per_file=10_000) == [100, 500, 500]
    assert allocate_budget([5000, 5000], total=10_000, per_file=3000) == [3000, 3000]
    assert allocate_budget([], total=100, per_file=10) == []


def test_first_file_does_not_consume_the_whole_budget():
    huge, small = "".join(_hunk(i) for i in range(100)), _hunk(1)
    limits = ContextLimits(max_diff_chars=2000, max_chars_per_file=10_000, max_files_per_commit=10,
                           max_listed_file_names=100)
    rendered = render_diffs([FileDiff("a_huge.tf", huge), FileDiff("b_small.tf", small)], [], limits)

    assert "FILE: b_small.tf\n```diff\n" + small + "```" in rendered.text
    assert rendered.diff_truncated
    assert rendered.files_truncated == ["a_huge.tf"]
    assert rendered.submitted_diff_chars <= limits.max_diff_chars
    assert rendered.original_diff_chars == len(huge) + len(small)
    assert "NOTE: patch truncated" in rendered.text


def test_per_file_cap():
    limits = ContextLimits(max_diff_chars=10_000, max_chars_per_file=300, max_files_per_commit=10,
                           max_listed_file_names=100)
    rendered = render_diffs([FileDiff("a.tf", _hunk(1, lines=30))], [], limits)
    assert rendered.submitted_diff_chars <= 300 and rendered.diff_truncated


def test_files_beyond_limit_are_listed_by_name():
    diffs = [FileDiff(f"f{i}.tf", _hunk(i, 1)) for i in range(5)]
    limits = ContextLimits(max_diff_chars=10_000, max_chars_per_file=10_000, max_files_per_commit=2,
                           max_listed_file_names=2)
    rendered = render_diffs(diffs, ["bin.tf"], limits)

    assert rendered.files_submitted == 2
    assert rendered.files_omitted == ["f2.tf", "f3.tf", "f4.tf"]
    assert rendered.diff_truncated
    assert "NOT SHOWN (limit of 2 patches per commit): f2.tf, f3.tf, ... and 1 more" in rendered.text
    assert "WITHOUT A TEXTUAL PATCH" in rendered.text and "bin.tf" in rendered.text
    assert "FILE: f2.tf" not in rendered.text


def test_identical_patch_files_are_named():
    rendered = render_diffs([FileDiff("a.tf", _hunk(1), also_in=["b.tf", "c.tf"])], [], BIG)
    assert "also applied to: b.tf, c.tf" in rendered.text


def test_commit_without_any_diff():
    rendered = render_diffs([], [], BIG)
    assert "no Terraform changes" in rendered.text
    assert rendered.original_diff_chars == 0 and not rendered.diff_truncated


def test_limits_must_be_positive():
    with pytest.raises(ValueError):
        ContextLimits.from_config({"max_diff_chars": 0, "max_chars_per_file": 1, "max_files_per_commit": 1,
                                   "max_listed_file_names": 1})
