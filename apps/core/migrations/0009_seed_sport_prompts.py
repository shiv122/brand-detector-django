"""Seed default SportPrompt rows for golf / cricket / football / nrl / generic.

Each prompt mirrors the user-provided RunPod prompt shape: request a JSON
object with sport-specific top-level fields plus universal `texts` and
`brands` arrays. The endpoint returns markdown-fenced JSON; the backend
strips the fence and parses it.
"""

from django.db import migrations


GOLF_PROMPT = (
    "This is a frame from a golf broadcast. Output ONLY a valid JSON object "
    "(no markdown fence, no commentary) with this shape:\n"
    "\n"
    "{\n"
    "  \"hole\": <int, current hole number, shown above par>,\n"
    "  \"par\": <int, par for current hole>,\n"
    "  \"texts\": [ <list every readable on-screen text string, verbatim, including "
    "the tournament name, leaderboard entries, player names, scores, hole "
    "indicators, captions, watermarks, and any other text. Be exhaustive — do "
    "not summarize.> ],\n"
    "  \"brands\": [ <only commercial/corporate logos and sponsors: companies, "
    "TV networks, watch makers, airlines, banks, governing bodies (EGA, PGA, "
    "R&A, DP World Tour), tournament title sponsors, product names> ]\n"
    "}\n"
    "\n"
    "Rules:\n"
    "- The texts array must include EVERY visible piece of text.\n"
    "- The brands array must contain ONLY companies / sponsors / governing "
    "bodies — never a player surname, score, hole number, or generic phrase "
    "like BIRDIE PUTT.\n"
    "- If unsure whether something is a brand, leave it out of brands (it will "
    "still appear in texts)."
)

CRICKET_PROMPT = (
    "This is a frame from a cricket broadcast. Output ONLY a valid JSON object "
    "(no markdown fence, no commentary) with this shape:\n"
    "\n"
    "{\n"
    "  \"score\": <string, combined scoreline as shown, e.g. \"245/6 (38.2)\">,\n"
    "  \"runs\": <int or null>,\n"
    "  \"wickets\": <int or null>,\n"
    "  \"overs\": <string or null, e.g. \"12.4\">,\n"
    "  \"batting_team\": <string or null>,\n"
    "  \"bowling_team\": <string or null>,\n"
    "  \"target\": <int or null, chase target if shown>,\n"
    "  \"texts\": [ <every readable on-screen text, verbatim and exhaustive> ],\n"
    "  \"brands\": [ <commercial/corporate logos and sponsors only> ]\n"
    "}\n"
    "\n"
    "Rules:\n"
    "- texts must include EVERY visible string.\n"
    "- brands must contain ONLY sponsors / companies / governing bodies — never "
    "a player name, team name, score, or generic phrase.\n"
    "- If unsure, leave it out of brands (it'll still appear in texts)."
)

FOOTBALL_PROMPT = (
    "This is a frame from a football (soccer) broadcast. Output ONLY a valid "
    "JSON object (no markdown fence, no commentary) with this shape:\n"
    "\n"
    "{\n"
    "  \"score\": <string, scoreline as shown, e.g. \"2-1\">,\n"
    "  \"home_team\": <string or null>,\n"
    "  \"away_team\": <string or null>,\n"
    "  \"home_score\": <int or null>,\n"
    "  \"away_score\": <int or null>,\n"
    "  \"minute\": <string or null, match minute, e.g. \"45+2\" or \"76'\">,\n"
    "  \"texts\": [ <every readable on-screen text, verbatim and exhaustive> ],\n"
    "  \"brands\": [ <commercial/corporate logos and sponsors only> ]\n"
    "}\n"
    "\n"
    "Rules:\n"
    "- texts must include EVERY visible string.\n"
    "- brands must contain ONLY sponsors / companies / governing bodies — never "
    "a player name, team name, score, or generic phrase.\n"
    "- If unsure, leave it out of brands."
)

NRL_PROMPT = (
    "This is a frame from an NRL (rugby league) broadcast. Output ONLY a valid "
    "JSON object (no markdown fence, no commentary) with this shape:\n"
    "\n"
    "{\n"
    "  \"score\": <string, scoreline as shown, e.g. \"12-6\">,\n"
    "  \"home_team\": <string or null>,\n"
    "  \"away_team\": <string or null>,\n"
    "  \"home_score\": <int or null>,\n"
    "  \"away_score\": <int or null>,\n"
    "  \"time\": <string or null, game clock as shown>,\n"
    "  \"half\": <string or null, half / period if shown>,\n"
    "  \"texts\": [ <every readable on-screen text, verbatim and exhaustive> ],\n"
    "  \"brands\": [ <commercial/corporate logos and sponsors only> ]\n"
    "}\n"
    "\n"
    "Rules:\n"
    "- texts must include EVERY visible string.\n"
    "- brands must contain ONLY sponsors / companies / governing bodies — never "
    "a player name, team name, score, or generic phrase.\n"
    "- If unsure, leave it out of brands."
)

GENERIC_PROMPT = (
    "This is a frame from a sports broadcast. Output ONLY a valid JSON object "
    "(no markdown fence, no commentary) with this shape:\n"
    "\n"
    "{\n"
    "  \"score\": <string or null, the visible scoreline as written>,\n"
    "  \"texts\": [ <every readable on-screen text, verbatim and exhaustive> ],\n"
    "  \"brands\": [ <commercial/corporate logos and sponsors only> ]\n"
    "}\n"
    "\n"
    "Rules:\n"
    "- texts must include EVERY visible string.\n"
    "- brands must contain ONLY sponsors / companies / governing bodies — never "
    "a player name, team name, score, or generic phrase.\n"
    "- If unsure, leave it out of brands."
)


SEEDS = [
    {
        "slug": "golf",
        "name": "Golf",
        "sport": "golf",
        "description": "Golf broadcasts — hole, par, leaderboard texts, brands.",
        "prompt": GOLF_PROMPT,
    },
    {
        "slug": "cricket",
        "name": "Cricket",
        "sport": "cricket",
        "description": "Cricket broadcasts — score, overs, wickets, brands.",
        "prompt": CRICKET_PROMPT,
    },
    {
        "slug": "football",
        "name": "Football (Soccer)",
        "sport": "football",
        "description": "Football (soccer) broadcasts — teams, scoreline, minute, brands.",
        "prompt": FOOTBALL_PROMPT,
    },
    {
        "slug": "nrl",
        "name": "NRL",
        "sport": "nrl",
        "description": "NRL (rugby league) broadcasts — teams, score, clock, brands.",
        "prompt": NRL_PROMPT,
    },
    {
        "slug": "generic",
        "name": "Generic sports",
        "sport": "generic",
        "description": "Any sport — score, texts, brands.",
        "prompt": GENERIC_PROMPT,
    },
]


def seed(apps, schema_editor):
    SportPrompt = apps.get_model("core", "SportPrompt")
    for s in SEEDS:
        if SportPrompt.objects.filter(slug=s["slug"]).exists():
            continue
        SportPrompt.objects.create(
            name=s["name"],
            slug=s["slug"],
            sport=s["sport"],
            description=s["description"],
            prompt=s["prompt"],
        )


def unseed(apps, schema_editor):
    SportPrompt = apps.get_model("core", "SportPrompt")
    for s in SEEDS:
        SportPrompt.objects.filter(slug=s["slug"]).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0008_sportprompt_delete_customocrtemplate"),
    ]

    operations = [
        migrations.RunPython(seed, unseed),
    ]
