from rsna2024_lumbar.preprocessing.catalog import iter_crop_jobs
from rsna2024_lumbar.preprocessing.constants import CROP_POLICY_CENTERED, LUMBAR_LEVELS


def test_skips_incomplete_study(mini_raw):
    jobs = iter_crop_jobs(
        mini_raw,
        crop_policy=CROP_POLICY_CENTERED,
        conditions=["spinal_canal_stenosis"],
    )
    studies = {j.study_id for j in jobs}
    assert studies == {1001, 1003}
    assert 1002 not in studies
    assert len(jobs) == 10
    assert {j.level for j in jobs if j.study_id == 1001} == set(LUMBAR_LEVELS)


def test_max_studies_is_a_prefix(mini_raw):
    jobs = iter_crop_jobs(
        mini_raw,
        crop_policy=CROP_POLICY_CENTERED,
        conditions=["spinal_canal_stenosis"],
        max_studies=1,
    )
    assert {j.study_id for j in jobs} == {1001}
