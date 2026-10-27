from lnp_rag.ingest.chunker import chunk_document, embedding_text, split_sentences
from lnp_rag.ingest.metadata import extract_authors, extract_doi, extract_title, extract_year
from lnp_rag.ingest.pdf_parser import Block, caption_label, parse_pdf, section_category
from lnp_rag.models import ChunkType

from .conftest import UPTAKE_PAPER, make_pdf


def test_caption_and_section_patterns():
    assert caption_label("Figure 2. Uptake kinetics") == ("figure_caption", "Figure 2")
    assert caption_label("Fig. 3A Flow cytometry") == ("figure_caption", "Figure 3A")
    assert caption_label("Fig. 5 β-Sitosterol modifies escape") == ("figure_caption", "Figure 5")
    assert caption_label("Table 1: Particle size") == ("table_caption", "Table 1")
    assert caption_label("Figure 2 shows that uptake") is None  # in-text reference, not a caption
    assert section_category("2.1. Materials and Methods") == "methods"
    assert section_category("RESULTS AND DISCUSSION") == "results"
    assert section_category("a b s t r a c t") == "abstract"
    assert section_category("STAR★METHODS") == "methods"
    assert section_category("Acknowledgements") == "backmatter"
    assert section_category("Cellular uptake study") is None


def test_parse_synthetic_pdf_structure(tmp_path):
    doc = parse_pdf(make_pdf(tmp_path / "p.pdf", UPTAKE_PAPER))
    headings = [b.text for b in doc.blocks if b.kind == "heading"]
    assert "3. Results" in headings and "References" in headings
    captions = [b for b in doc.blocks if b.kind == "figure_caption"]
    assert len(captions) == 1 and captions[0].label == "Figure 2"
    results = [b for b in doc.blocks if b.kind == "paragraph" and b.category == "results"]
    assert results and "time-dependent" in results[0].text
    assert all(b.page == 1 for b in doc.blocks)


def test_metadata_extraction(tmp_path):
    doc = parse_pdf(make_pdf(tmp_path / "p.pdf", UPTAKE_PAPER))
    title, source = extract_title(doc)
    assert title == UPTAKE_PAPER["title"] and source == "pdf"
    assert extract_doi(doc) == "10.1234/jnano.2023.0042"
    assert extract_year(doc) == 2023
    assert extract_authors(doc, title) == ["Jane Smith", "Rahul Mehta", "Wei Chen"]


def test_chunking_keeps_structure(tmp_path):
    doc = parse_pdf(make_pdf(tmp_path / "p.pdf", UPTAKE_PAPER))
    chunks, abstract = chunk_document("abc", doc.blocks, target_words=60, overlap_words=10, min_words=5)
    assert abstract.startswith("Lipid nanoparticles (LNPs)")
    assert chunks[0].chunk_type == ChunkType.ABSTRACT
    types = {c.chunk_type for c in chunks}
    assert ChunkType.FIGURE_CAPTION in types
    # References are not indexed
    assert not any("unrelated cited paper" in c.text for c in chunks)
    # ordinals / ids are sequential and unique
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    caption = next(c for c in chunks if c.chunk_type == ChunkType.FIGURE_CAPTION)
    assert caption.label == "Figure 2" and caption.features.n_timepoints == 5
    assert embedding_text(caption, "T").startswith("T\nFigure 2: ")


def test_windowing_overlap_and_section_boundaries():
    sent = "Uptake was measured in cells at several doses and times. "
    blocks = [
        Block(sent * 12, 3, "paragraph", 10, False, (0, 0, 1, 1), section="3. Results", category="results"),
        Block(sent * 2, 4, "paragraph", 10, False, (0, 0, 1, 1), section="4. Discussion", category="discussion"),
    ]
    chunks, _ = chunk_document("p", blocks, target_words=40, overlap_words=10, min_words=5)
    results = [c for c in chunks if c.section == "3. Results"]
    assert len(results) >= 3
    # consecutive windows overlap: the next window starts with the previous one's last sentence
    last_sentence = split_sentences(results[0].text)[-1]
    assert results[1].text.startswith(last_sentence)
    # no chunk mixes sections
    assert all(c.section in ("3. Results", "4. Discussion") for c in chunks)
    assert any(c.section == "4. Discussion" for c in chunks)


def test_sentence_splitter_respects_abbreviations():
    s = split_sentences("Uptake rose (Fig. 2) at 0.5 h. LNPs et al. showed this. Next sentence.")
    assert s == ["Uptake rose (Fig. 2) at 0.5 h.", "LNPs et al. showed this.", "Next sentence."]
