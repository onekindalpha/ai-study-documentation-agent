# Study Documentation Automation Ai Agent

**Language:** English | [한국어](./README.ko.md)

AI Study Documentation Agent is a study documentation pipeline that turns scattered learning records into structured technical writing.

During lectures, coding labs, and debugging sessions, useful learning evidence is often fragmented across screenshots, URLs, rough notes, error logs, and Q&A. This project organizes those records into session-based evidence and converts them into reusable technical notes, troubleshooting records, and problem-solving blog drafts.

The project focuses on the flow from **study evidence** to **context reconstruction** to **technical writing output**.

---

## Demo

- Live Demo: https://huggingface.co/spaces/onekindalpha/study-documentation-ai-agent

https://github.com/user-attachments/assets/1af3dd29-a7b9-49ec-b49d-a37b8e9443b2

The guided video shows how to add a learning source, start article generation, wait for source collection and analysis, review the generated Markdown, and prepare the learning record for blog publishing.

## Overview

This project is designed for learners and developers who want to preserve the reasoning process behind their study and practice.

Instead of treating screenshots, notes, errors, and questions as separate fragments, the system groups them into a connected learning session. That session can then be used to generate structured technical documentation.

The goal is not simple note storage.  
The goal is to turn real learning traces into reusable documentation.

---

## What It Does

This project works as a capture-to-writing workflow for technical learning records.

It helps users:

- collect screenshots, course URLs, notes, error logs, and Q&A records
- organize learning evidence by capture or session
- classify learning URLs and route them to site-aware extractors
- collect source context from public pages, rendered sites, course structures, and YouTube transcripts
- build a run-scoped Source Graph and Source Pack before writing
- interpret screenshot evidence with a vision-capable LLM provider
- preserve Q&A history from the learning process
- generate Medium-style technical blog drafts focused on problem recognition, cause analysis, action, validation, and outcome
- block stale-topic contamination and unsupported URL-only drafts with quality and topic guards
- copy the generated output as Markdown

The goal is not to replace the learner's judgment. The goal is to preserve the reasoning path behind learning and debugging so it can be reconstructed later as technical documentation.

---

## Key Features

- Screenshot-based learning evidence reconstruction
- Universal learning-source URL detection and extractor routing
- YouTube transcript, metadata, chapter, and fallback enrichment
- Site-aware extraction for Agent Academy, WikiDocs, Notion, Oopy, AI Skills Navigator, and generic public pages
- Protected-course partial reports for login- or enrollment-gated sources
- Run-scoped Source Graph, Source Pack, trace, and quality-gate artifacts
- Session-based capture timeline
- Q&A log and tutor-style answer record
- Image evidence builder with vision/fallback handling
- Problem map and decision map generation
- Article brief and section plan generation
- Medium-style Markdown draft generation
- Source-first, seed-URL, topic-lock, article-policy, and voice guards for final draft validation
- Debug tabs for collector reports, captures, Q&A, evidence, maps, plans, briefs, and critic results
- Markdown copy and draft reset support

---

## Architecture

```mermaid
flowchart TB
    IMG["Lecture Screenshots"] --> CAP["Session Capture"]
    TXT["Raw Text / Lab Notes"] --> CAP
    ERR["Error Logs"] --> CAP
    URL["Course / Video / Lab URLs"] --> DETECT["URL Detector / Collection Plan"]
    QA["Q&A Records"] --> QALOG["Q&A / Tutor Log"]

    CAP --> RECORDS["Local Study Records"]
    QALOG --> RECORDS

    DETECT --> YT["YouTube Extractor"]
    DETECT --> SITE["Site-aware Extractors"]
    DETECT --> WEB["Generic / Rendered Web Extractor"]
    DETECT --> FALLBACK["Protected-course Partial Report"]

    RECORDS --> EVID["Image Evidence Builder"]
    YT --> GRAPH["Run-scoped Source Graph"]
    SITE --> GRAPH
    WEB --> GRAPH
    FALLBACK --> GRAPH
    GRAPH --> QUALITY["Quality Gate / Evidence Ranking"]
    QUALITY --> PACK["Source Pack + Trace Artifacts"]

    VISION["Groq Vision Runtime"] --> EVID
    EVID --> MAP["Problem Map / Decision Map"]
    PACK --> MAP
    RECORDS --> MAP

    MAP --> PLAN["Article Brief / Section Plan"]
    PLAN --> GEN["Medium-style Markdown Generator"]
    GROQ["Groq Text Runtime"] --> GEN

    GEN --> GUARD["Seed URL / Topic / Policy / Voice Guards"]
    GUARD --> UI["Web UI<br/>Draft + Debug Artifacts"]
    UI --> COPY["Copy Markdown"]
    UI --> RESET["Reset Draft / Inputs"]
```

---

## System Flow

```mermaid
flowchart LR
    A["Add screenshots, URLs, notes, or Q&A"] --> B["Collect source context"]
    B --> C["Build Source Graph and run quality gate"]
    C --> D["Build image and text evidence"]
    D --> E["Create maps, brief, and section plan"]
    E --> F["Generate and validate Markdown draft"]
    F --> G["Review, copy, or reset output"]
```

---

## Implementation Notes

- **Capture-first workflow**: screenshots, raw text, memo fields, source URLs, and Q&A records are treated as learning evidence rather than isolated inputs.
- **Session timeline**: the service can group multiple captures and Q&A logs into a single learning session before generating a draft.
- **Source collection**: a URL detector selects a YouTube, site-aware, protected-course, or generic-web extractor. The current extractor set covers YouTube, Microsoft Agent Academy, WikiDocs, Notion, Oopy, AI Skills Navigator, Inflearn/Udemy partial access, and ordinary public pages.
- **Run isolation**: each URL generation request receives its own run ID and output directory so an earlier Source Pack cannot silently leak into the current article.
- **Source Graph and quality gate**: collectors normalize titles, text, links, videos, chapters, labs, code, and access limitations into a Source Graph. Quality gates decide whether the evidence is sufficient for article generation.
- **Vision-assisted evidence extraction**: screenshots are interpreted as visual learning evidence and mapped into captions, visible evidence, roles, problem signals, and technical entities.
- **Problem reconstruction**: evidence is organized into a problem map, decision map, article brief, and section plan before final article generation.
- **Grounded draft generation**: the generated article uses captured evidence, collected source context, Q&A logs, and user notes as inputs.
- **Final draft guard**: source-first checks, seed-URL matching, topic locks, contamination detection, evidence coverage, and article-policy checks prevent stale or unsupported drafts from being presented as final output.
- **Fallback behavior**: when source collection or LLM generation is unavailable, the service returns a safer fallback note instead of fabricating unsupported details.

---

## Tech Stack

- Backend: Python standard library HTTP server
- Frontend: HTML, CSS, JavaScript single-page UI
- LLM runtime: Groq text generation and Groq vision; provider-routing configuration exists, while the deployed client path currently uses Groq
- Source collection: Playwright, Crawl4AI, Trafilatura, `youtube-transcript-api`, `yt-dlp`, and site-specific extractors
- Storage: local notes, sessions, captures, run artifacts, Source Graphs, Source Packs, and collector traces
- Output: Markdown draft generation

---

## Project Status

This project is a portfolio-stage prototype focused on converting real study evidence into reusable technical documentation.

The current implementation covers session capture, site-aware URL collection, Source Graph normalization, quality gating, evidence reconstruction, guarded draft generation, and collector/debug inspection.

The project is still being improved, especially around backend modularization, provider abstraction, asynchronous generation, persistent storage, export options, test coverage, and public demo stability.

---

## Roadmap

Planned improvements include:

- separating the backend into smaller modules
- improving browser-based capture flow
- adding stronger export options for Markdown and Notion
- adding asynchronous jobs, progress streaming, transcript/source caching, and shorter generation latency
- completing a provider-independent LLM client layer
- expanding regression tests for collectors, evidence processing, topic guards, and draft generation
- stabilizing public demo resources

---

## Development Notes

Local setup, environment variables, API routes, runtime data paths, and deployment notes are separated into [DEVELOPMENT.md](./DEVELOPMENT.md).

---

## Portfolio Context

This repository is positioned as an AI service / documentation workflow portfolio project.

It shows:

- designing a session-based capture workflow
- collecting source context from learning materials
- designing site-aware collectors, Source Graphs, quality gates, and run-isolated artifacts
- turning screenshots and notes into structured learning evidence
- generating problem-solving technical writing from fragmented study records
- handling incomplete, protected, or weak learning sources safely
- exposing a browser-based UI for capture, generation, debugging, copy, and reset workflows

The project is connected with other AI/backend portfolio work:

- Battery RUL AI Inference System: model inference, dashboard, and deployment
- Battery Technical Document RAG Assistant: technical document retrieval and grounded answer generation
- AWS 3-Tier Runbook AI Agent: infrastructure documentation search and troubleshooting support

---

## Honest Scope

This project does:

- organize study captures and Q&A records
- collect and quality-check public or partially accessible learning-source context
- interpret screenshots as learning evidence
- generate Markdown-based study notes and technical article drafts
- support portfolio-style problem-solving documentation

It does not:

- automatically access protected course pages without provided context
- guarantee complete transcripts when captions are disabled or unavailable
- guarantee correctness when source material is incomplete
- replace manual technical review before publishing
- operate as a general-purpose autonomous browser agent
