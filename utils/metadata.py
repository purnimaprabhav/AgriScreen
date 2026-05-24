from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ChunkMetadata:
    """
    Metadata attached to every chunk that enters the vector index.
    Every field is used for filtered retrieval later in retriever.py.
    """
    doc_type    : str            # factsheet | news | report | financials | funding
    source_file : str            # original filename, e.g. aquagrow_solutions_ltd_factsheet.txt

    # Set for factsheets and news; None for reports
    company     : Optional[str] = None   # full company name, e.g. "AquaGrow Solutions Ltd"
    company_id  : Optional[str] = None   # e.g. "AG-004" — links to CSV rows

    # Factsheet-specific
    section     : Optional[str] = None   # e.g. "TECHNOLOGY", "ESG & SUSTAINABILITY"
    is_scoring_inputs: bool     = False  # True only for the structured SCORING INPUTS block

    # News-specific
    news_id     : Optional[str] = None   # e.g. "NEWS-006"
    date        : Optional[str] = None   # e.g. "15 March 2026"


@dataclass
class Chunk:
    """
    A single unit that gets embedded and stored in the vector index.
    text     → what gets embedded
    metadata → what gets stored alongside the vector for filtering
    """
    text     : str
    metadata : ChunkMetadata

    def to_dict(self) -> dict:
        return {
            "text"    : self.text,
            "metadata": asdict(self.metadata)
        }