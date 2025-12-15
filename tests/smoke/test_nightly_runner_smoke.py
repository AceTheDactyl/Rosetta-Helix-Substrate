from pathlib import Path


def test_nightly_runner_minimal(tmp_path):
    from nightly_training_runner import run_nightly_training

    outdir = tmp_path / "nightly"
    res = run_nightly_training(output_dir=str(outdir), force_runs=1)

    assert res["status"] in {"success", "failed"}
    assert "coherence" in res and "training" in res and "summary" in res
    assert (outdir / "training_results.json").exists()
    assert (outdir / "TRAINING_SUMMARY.md").exists()
    assert (outdir / "visualizer_data.json").exists()

