# Source-Pack Collection Profile

## Goal

The collector accepts a representative source URL and builds a source pack for later problem-solving article generation.

Representative URLs can be:

- AI Skills Navigator player URLs
- article or documentation pages
- YouTube or other video pages
- Oopy/Notion-style pages
- WikiDocs/book/chapter pages
- lab, exercise, or instruction pages

Udemy is intentionally excluded from automatic collection when Cloudflare,
login, or course-entitlement gates are shown. In that case, collect a manual
source pack from the normal browser by copying the course title, curriculum,
lecture transcript/description, and resource links.

The collector should gather visible text, headings, links, lesson candidates, video candidates, and lab/exercise/instruction candidates from the current source, then follow bounded related source links.

## AI Skills Navigator Start Point

Use the final playlist player URL after the user manually selects one playlist card.

Expected URL shape:

```text
https://aiskillsnavigator.microsoft.com/player?playlistId=<playlist-id>
```

Do not start from the event page, card grid, recommendation list, or search results.

## Manual steps

The user does these steps:

1. Log in if needed.
2. Choose one playlist card.
3. Open the playlist player page.
4. Copy the final player URL.
5. Give that URL to the collector.

The collector does not automate login and does not choose playlist cards.

## Collector command

```bash
cd ~/Developer/study-capture-copilot-deploy
python tools/collect_source_pack.py \
  "https://aiskillsnavigator.microsoft.com/player?playlistId=<playlist-id>" \
  --follow-labs \
  --user-data-dir data/browser_profiles/ai-skills-navigator
```

Generic URL examples:

```bash
python tools/collect_source_pack.py "https://youtu.be/<video-id>" --no-manual-pause
python tools/collect_source_pack.py "https://example.oopy.io/page" --no-manual-pause
python tools/collect_source_pack.py "https://wikidocs.net/book/13314" --no-manual-pause
```

To reuse login sessions, use one of:

```bash
python tools/collect_source_pack.py "<url>" --user-data-dir data/browser_profiles/source-collector
python tools/collect_source_pack.py "<url>" --storage-state data/browser_state/source-collector.json
```

If Playwright is not installed:

```bash
python -m pip install playwright
python -m playwright install chromium
```

## Collector behavior

The collector:

- opens the provided representative URL
- pauses for manual login when redirected to Microsoft login
- pauses for manual confirmation before collecting
- saves the primary page snapshot before clicking lesson/lab navigation items
- expands Summary, Transcript, Details, Show more, and similar sections when visible
- for AI Skills Navigator player pages, opens player navigation and clicks visible lesson/lab tree items
- extracts visible page text
- extracts headings
- extracts links, video candidates, lesson candidates, and lab candidates
- follows bounded same-origin article/docs/lesson links
- follows Lab / Exercise / Instructions links when `--follow-labs` is used
- saves timestamped `.md`, `.json`, and `.report.md` files under `data/source_packs`
- prints collection stats

## Output files

The collector writes:

- `data/source_packs/YYYYMMDD_HHMMSS_<title>.md`
- `data/source_packs/YYYYMMDD_HHMMSS_<title>.json`
- `data/source_packs/YYYYMMDD_HHMMSS_<title>.report.md`
