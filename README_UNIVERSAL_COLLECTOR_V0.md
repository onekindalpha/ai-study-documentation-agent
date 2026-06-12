# Universal Learning Source Collector v0

This branch adds a first isolated collector skeleton for learning URLs.

The collector does not generate Medium articles yet. It only creates:

- `source_graph.json`
- `collection_report.json`
- `source_pack.md`
- `trace.jsonl`
- optional screenshots for rendered pages

## Run

```bash
python tools/universal_learning_collector.py "https://youtu.be/d2X38zE7VsU?si=mWMMJgacxsJ3P_H7" --max-pages 8
python tools/universal_learning_collector.py "https://microsoft.github.io/agent-academy/" --max-pages 8
```

For rendered pages such as Oopy, Inflearn, or Udemy:

```bash
python tools/universal_learning_collector.py "https://0chnxxx.oopy.io/126c9413-0950-8096-bd5c-c7d2069d6294" --visible --max-pages 5 --max-depth 1
```

## Smoke test

```bash
python tools/run_universal_collector_smoke.py
```

## Current scope

Implemented v0 extractors:

- YouTube transcript extractor
- Agent Academy course extractor
- WikiDocs book extractor
- Oopy / Notion-like child-page extractor
- Protected course partial extractor for Inflearn/Udemy
- Generic web fallback
- AI Skills placeholder adapter

## Important rule

Medium article generation is intentionally not connected yet. A URL must first produce acceptable collection evidence.
