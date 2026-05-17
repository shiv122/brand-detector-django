"""
Pre-defined OCR formatting templates.

Each template declares:
  - key                 stable identifier
  - label               UI display name
  - description         short blurb for the UI
  - mode                "none" | "regex" | "llm"
  - system_prompt       (llm) instruction sent to Gemini
  - schema              (llm) Pydantic model — Gemini returns JSON matching it
  - pattern             (regex) single regex; first match's named groups are returned
  - multimodal          (llm) include the cropped image alongside OCR text
  - supports_custom_prompt   allow user to override system_prompt at run time
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field


class ScoreGeneric(BaseModel):
    ocr_brands: List[str] = Field(
        default_factory=list,
        description=(
            "Sponsor/brand names visible in the image (hoardings, jerseys, "
            "logos, scoreboard sponsors). Empty list if none."
        ),
    )
    score: Optional[str] = Field(
        None,
        description="Scoreline as written, e.g. '2-1', '24:21', 'HOME 12 AWAY 7'.",
    )
    home: Optional[str] = Field(None, description="Home team name or label")
    away: Optional[str] = Field(None, description="Away team name or label")
    home_score: Optional[str] = Field(None, description="Home score as written")
    away_score: Optional[str] = Field(None, description="Away score as written")
    period: Optional[str] = Field(None, description="Period/quarter/half if shown")


class GolfHole(BaseModel):
    ocr_brands: List[str] = Field(
        default_factory=list,
        description=(
            "Sponsor/brand names visible anywhere in the frame — green-side "
            "hoardings (e.g. 'EMIRATES', 'NAKHEEL'), broadcaster bug sponsors, "
            "tournament title sponsors, etc. Deduplicate, uppercase preserved."
        ),
    )
    hole: Optional[int] = Field(
        None,
        description=(
            "The CURRENT hole number being played (1-18). This is the large "
            "standalone number in the dedicated 'current hole' on-screen "
            "graphic — NOT any number in a leaderboard."
        ),
    )
    par: Optional[int] = Field(
        None,
        description=(
            "Par for the current hole (typically 3, 4, or 5). Usually shown "
            "as 'PAR N' directly under or beside the hole number."
        ),
    )
    score: Optional[str] = Field(
        None,
        description=(
            "The featured player's score on this hole as written in the HUD "
            "(e.g. 'PAR', '-1', '+2', 'E'). NOT the tournament-overall score "
            "from the leaderboard."
        ),
    )
    player: Optional[str] = Field(
        None,
        description="Featured player surname from the HUD (e.g. 'HATTON').",
    )
    yardage: Optional[int] = Field(
        None, description="Hole yardage if shown (e.g. '420 YDS')."
    )
    shot_context: Optional[str] = Field(
        None,
        description=(
            "Shot-context text from the HUD if any, e.g. 'FOR BIRDIE', "
            "'FOR PAR', 'TO WIN'."
        ),
    )


class CricketScore(BaseModel):
    ocr_brands: List[str] = Field(
        default_factory=list,
        description=(
            "Sponsor/brand names visible anywhere — boundary hoardings, bat/jersey "
            "logos, scoreboard sponsors, broadcaster bugs. Deduplicate."
        ),
    )
    score: Optional[str] = Field(
        None,
        description=(
            "Combined scoreline as written, e.g. '245/6 (38.2)' or '180-4'."
        ),
    )
    runs: Optional[int] = None
    wickets: Optional[int] = None
    overs: Optional[str] = Field(None, description="Overs as shown, e.g. '12.4'")
    batting_team: Optional[str] = None
    bowling_team: Optional[str] = None
    target: Optional[int] = None


class FootballScore(BaseModel):
    ocr_brands: List[str] = Field(
        default_factory=list,
        description=(
            "Sponsor/brand names visible — pitch-side LED boards, jersey "
            "sponsors, broadcaster bugs, competition sponsors. Deduplicate."
        ),
    )
    score: Optional[str] = Field(
        None, description="Scoreline as written, e.g. '2-1' or 'ARS 2 - 1 LIV'."
    )
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    minute: Optional[str] = Field(None, description="Match minute, e.g. '45+2' or '76\\'.")


class NrlScore(BaseModel):
    ocr_brands: List[str] = Field(
        default_factory=list,
        description=(
            "Sponsor/brand names visible — ground-side boards, jersey sponsors, "
            "scoreboard sponsors, broadcaster bugs. Deduplicate."
        ),
    )
    score: Optional[str] = Field(
        None, description="Scoreline as written, e.g. '12-6' or 'PEN 24 - 18 SOU'."
    )
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    time: Optional[str] = Field(None, description="Game clock as shown")
    half: Optional[str] = Field(None, description="Half/period if shown")


class CustomFreeform(BaseModel):
    """Wrapper for custom prompts — Gemini returns whatever shape the prompt asks for."""

    result: Any = Field(None, description="Free-form result for custom prompts")


@dataclass
class PromptTemplate:
    key: str
    label: str
    description: str
    mode: str
    system_prompt: str = ""
    schema: Optional[Type[BaseModel]] = None
    pattern: Optional[re.Pattern] = None
    multimodal: bool = False
    supports_custom_prompt: bool = True

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "mode": self.mode,
            "multimodal": self.multimodal,
            "supports_custom_prompt": self.supports_custom_prompt,
            "schema": self.schema.model_json_schema() if self.schema else None,
        }


_BASE_INSTRUCTION = (
    "You are an OCR post-processor. You will be given raw OCR text "
    "extracted from a sports broadcast or graphic. Extract the requested "
    "fields and return JSON matching the provided schema. Use null for any "
    "field that is not clearly present in the text. Do not invent values. "
    "If the text is empty or unrelated, return null fields (and an empty "
    "list for list fields)."
)


_BRANDS_GUIDANCE = (
    "For `ocr_brands`: include any text that reads as a sponsor or brand "
    "name — company names on boundary/pitch-side hoardings, jersey/bib "
    "sponsors, broadcaster bugs, tournament title sponsors, etc. Exclude "
    "score numbers, player names, team names, generic words ('PAR', "
    "'HOLE', 'GOAL', 'WICKET'), and time/clock text. Deduplicate. Preserve "
    "the casing as it appears on screen (typically uppercase)."
)


_TEMPLATES: Dict[str, PromptTemplate] = {
    "raw": PromptTemplate(
        key="raw",
        label="Raw text",
        description="Return OCR text as-is with no formatting.",
        mode="none",
        supports_custom_prompt=False,
    ),
    "score_generic": PromptTemplate(
        key="score_generic",
        label="Sports score (generic)",
        description="Extract a generic team-vs-team scoreline + brands.",
        mode="llm",
        system_prompt=(
            _BASE_INSTRUCTION
            + " Extract a generic team-vs-team scoreline (home/away with scores) "
            "AND populate `ocr_brands` with any sponsor/brand text. "
            "`score` is the scoreline as written (e.g. '2-1'). "
            "`home_score`/`away_score` are the same numbers as written.\n\n"
            + _BRANDS_GUIDANCE
        ),
        schema=ScoreGeneric,
    ),
    "golf_hole": PromptTemplate(
        key="golf_hole",
        label="Golf hole number",
        description=(
            "Read the current hole number, par, featured player, and shot "
            "context from the on-screen 'current hole' HUD."
        ),
        mode="llm",
        multimodal=True,
        system_prompt=(
            _BASE_INSTRUCTION
            + "\n\nThe image is a frame from a golf broadcast. There is "
            "typically a 'current hole' HUD (often in a top corner) shaped "
            "roughly like this:\n"
            "  ┌──────────────────────────────────┐\n"
            "  │  N    [+]  PLAYER_NAME    SCORE  │\n"
            "  │  PAR M       FOR BIRDIE          │\n"
            "  └──────────────────────────────────┘\n"
            "where:\n"
            "  • N is a LARGE standalone integer 1-18 — this is the `hole`.\n"
            "  • 'PAR M' (M is 3, 4, or 5) is the hole's `par`.\n"
            "  • PLAYER_NAME (often surname only, e.g. 'HATTON') is `player`.\n"
            "  • SCORE next to the player (e.g. 'PAR', '-1', '+2', 'E') is "
            "`player_score_to_par`.\n"
            "  • Phrases like 'FOR BIRDIE', 'FOR PAR', 'TO WIN' are "
            "`shot_context`.\n\n"
            "A scrolling LEADERBOARD may also be visible (a list of players "
            "with ranks, scores, and holes played). IGNORE the leaderboard "
            "when filling these fields — its left column is rank/position "
            "(1, 2, 3, …), NOT hole numbers, and its right column is "
            "holes-played, NOT par. Only read from the dedicated current-hole "
            "HUD.\n\n"
            "If no current-hole HUD is visible, return null for every "
            "hole/par/player/score/yardage/shot_context field. Do not invent "
            "values. Still fill `ocr_brands` if any sponsor text is visible "
            "elsewhere in the frame.\n\n"
            + _BRANDS_GUIDANCE
        ),
        schema=GolfHole,
    ),
    "cricket_score": PromptTemplate(
        key="cricket_score",
        label="Cricket score",
        description="Extract score / runs / wickets / overs + brands.",
        mode="llm",
        system_prompt=(
            _BASE_INSTRUCTION
            + " Cricket scoreboard. Extract:\n"
            "  • `score`: the combined scoreline as written (e.g. '245/6 (38.2)').\n"
            "  • `runs`, `wickets`, `overs` ('overs' is a string like '12.4'), "
            "and `batting_team`/`bowling_team` if labelled.\n"
            "  • `target` if a chase target is shown.\n"
            "  • `ocr_brands`: see below.\n\n"
            + _BRANDS_GUIDANCE
        ),
        schema=CricketScore,
    ),
    "football_score": PromptTemplate(
        key="football_score",
        label="Football score",
        description="Extract teams / scoreline / minute + brands.",
        mode="llm",
        system_prompt=(
            _BASE_INSTRUCTION
            + " Football (soccer) scoreboard. Extract:\n"
            "  • `score`: the scoreline as written (e.g. '2-1').\n"
            "  • `home_team`, `away_team`, integer `home_score`/`away_score`.\n"
            "  • `minute`: match minute string if shown.\n"
            "  • `ocr_brands`: see below.\n\n"
            + _BRANDS_GUIDANCE
        ),
        schema=FootballScore,
    ),
    "nrl_score": PromptTemplate(
        key="nrl_score",
        label="NRL score",
        description="Extract NRL teams / score / clock + brands.",
        mode="llm",
        system_prompt=(
            _BASE_INSTRUCTION
            + " NRL (rugby league) scoreboard. Extract:\n"
            "  • `score`: the scoreline as written (e.g. '12-6').\n"
            "  • `home_team`, `away_team`, integer `home_score`/`away_score`.\n"
            "  • `time`: game clock as shown; `half`: half/period if shown.\n"
            "  • `ocr_brands`: see below.\n\n"
            + _BRANDS_GUIDANCE
        ),
        schema=NrlScore,
    ),
    "time_clock": PromptTemplate(
        key="time_clock",
        label="Game clock (regex)",
        description="Match game-clock patterns like '12:34' or 'Q1 12:34'.",
        mode="regex",
        pattern=re.compile(
            r"(?P<period>Q[1-4]|H[12])?\s*(?P<clock>\d{1,2}:\d{2})",
            re.IGNORECASE,
        ),
        supports_custom_prompt=False,
    ),
    "custom": PromptTemplate(
        key="custom",
        label="Custom prompt",
        description="Free-form prompt — return whatever shape the prompt asks for.",
        mode="llm",
        system_prompt=(
            _BASE_INSTRUCTION
            + " Follow the user's custom instructions to shape the JSON output. "
            "Place the result under the `result` key."
        ),
        schema=CustomFreeform,
    ),
}


def list_templates() -> List[PromptTemplate]:
    return list(_TEMPLATES.values())


def get_template(key: str) -> Optional[PromptTemplate]:
    return _TEMPLATES.get(key)


def public_registry() -> List[Dict[str, Any]]:
    return [t.to_public_dict() for t in _TEMPLATES.values()]
