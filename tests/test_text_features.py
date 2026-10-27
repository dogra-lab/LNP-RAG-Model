from lnp_rag.ingest.cleaning import join_lines, normalize_text
from lnp_rag.ingest.features import (
    classify_study_type,
    detect_features,
    extract_timepoints,
    looks_like_review,
    query_intents,
)
from lnp_rag.models import ChunkFeatures, StudyType


def test_normalize_ligatures_hyphenation_and_whitespace():
    assert normalize_text("efﬁcient   uptake") == "efficient uptake"
    assert normalize_text("nano-\nparticle") == "nanoparticle"
    # capitalised continuation is a real hyphenated name
    assert normalize_text("DLin-\nMC3") == "DLin-MC3"
    assert normalize_text("5 µg") == "5 μg"  # micro sign -> greek mu
    assert join_lines("line one\nline two") == "line one line two"


def test_extract_timepoints_lists_ranges_and_days():
    assert extract_timepoints("at 0.5, 1, 2, 4 and 24 h") == ["0.5 h", "1 h", "2 h", "4 h", "24 h"]
    assert extract_timepoints("incubated for 30 min or 2 hours") == ["30 min", "2 h"]
    assert extract_timepoints("on day 1, 3 and 7") == ["1 d", "3 d", "7 d"]
    assert extract_timepoints("no time information here") == []


def test_detect_features_on_quantitative_uptake_text():
    f = detect_features(
        "Flow cytometry showed MFI of 1.8 ± 0.2 x 10^3 at 0.5 h and 2.4 ± 0.3 x 10^4 at 24 h "
        "(n = 3, p < 0.01) in HepG2 cells treated with 2 μg/mL mRNA."
    )
    assert f.has_quantitative and f.has_dose and f.has_uptake_terms and f.mentions_in_vitro
    assert f.n_timepoints == 2
    assert not f.mentions_in_vivo


def test_in_vivo_detection():
    f = detect_features("C57BL/6 mice were injected intravenously with LNPs; biodistribution was imaged.")
    assert f.mentions_in_vivo and not f.mentions_in_vitro


def test_study_type_ignores_introduction_mentions():
    vitro = ChunkFeatures(mentions_in_vitro=True)
    vivo = ChunkFeatures(mentions_in_vivo=True)
    rows = [("methods", vitro), ("results", vitro), ("introduction", vivo), ("introduction", vivo)]
    label, _ = classify_study_type(rows)
    assert label == StudyType.IN_VITRO
    rows += [("figure", vivo), ("methods", vivo)]
    assert classify_study_type(rows)[0] == StudyType.BOTH
    assert classify_study_type([])[0] == StudyType.UNKNOWN


def test_review_detection():
    assert looks_like_review("Recent advances in LNP delivery", "In this review we summarize...", [])
    assert not looks_like_review(
        "LNP uptake kinetics", "We formulated LNPs and measured uptake.", ["2. Materials and Methods", "3. Results"]
    )


def test_query_intents():
    intents = query_intents("quantitative uptake histograms over time at different doses")
    assert intents == {"quantitative": True, "multi_timepoint": True, "dose": True, "figure": True}
    assert not any(query_intents("ionizable lipid structure").values())


def test_feature_regexes_are_linear_on_pathological_text():
    import time

    # Long number lists without a unit (tables) and long digit runs used to backtrack
    # exponentially / quadratically and hang ingestion.
    texts = [
        ", ".join(str(i) for i in range(400)) + " x",
        " ,  ".join(str(i) for i in range(400)) + " and",
        "1" * 50_000,
        "0." + "5" * 50_000 + " h",
    ]
    for text in texts:
        t0 = time.perf_counter()
        detect_features(text)
        assert time.perf_counter() - t0 < 1.0, text[:40]
    assert extract_timepoints("at 1, 2, and 24 h") == ["1 h", "2 h", "24 h"]
