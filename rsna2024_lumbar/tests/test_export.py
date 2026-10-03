from PIL import Image

from rsna2024_lumbar.preprocessing.constants import CROP_POLICY_CENTERED
from rsna2024_lumbar.preprocessing.export import export_crops, png_relpath
from rsna2024_lumbar.preprocessing.catalog import iter_crop_jobs


def test_export_writes_png_manifest_meta(mini_raw, tmp_path):
    out = tmp_path / "centered"
    result = export_crops(
        mini_raw,
        out,
        crop_policy=CROP_POLICY_CENTERED,
        conditions=["spinal_canal_stenosis"],
    )
    assert result["exported"] == 10
    assert (out / "manifest.csv").is_file()
    assert (out / "export_meta.json").is_file()
    sample = out / "spinal_canal_stenosis" / "Normal_Mild" / "study_1001_l1_l2.png"
    assert sample.is_file()
    im = Image.open(sample)
    assert im.size == (64, 64)
    moderate = out / "spinal_canal_stenosis" / "Moderate" / "study_1003_l4_l5.png"
    assert moderate.is_file()


def test_skip_existing_resumes(mini_raw, tmp_path):
    out = tmp_path / "centered"
    export_crops(
        mini_raw, out, crop_policy=CROP_POLICY_CENTERED, conditions=["spinal_canal_stenosis"]
    )
    again = export_crops(
        mini_raw,
        out,
        crop_policy=CROP_POLICY_CENTERED,
        conditions=["spinal_canal_stenosis"],
        skip_existing=True,
    )
    assert again["exported"] == 0
    assert again["skipped"] == 10


def test_png_path_contract(mini_raw):
    job = iter_crop_jobs(
        mini_raw,
        crop_policy=CROP_POLICY_CENTERED,
        conditions=["spinal_canal_stenosis"],
        max_studies=1,
    )[0]
    rel = png_relpath(job)
    assert rel.parts[0] == job.condition
    assert rel.name == f"study_{job.study_id}_{job.level}.png"
