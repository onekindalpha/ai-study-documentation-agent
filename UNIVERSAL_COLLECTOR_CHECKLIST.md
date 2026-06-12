# Universal Learning Source Collector Checklist

## Purpose

This checklist tracks the development progress of the Universal Learning Source Collector.

Goal:

> A user enters a representative learning URL.  
> The app collects the learning content, including text, transcript, images, code, lesson structure, child pages, lab/exercise links, and related references.  
> The collected evidence is then used to generate a problem-solving Medium article from the perspective of a learner who studied the material, struggled with complex parts, understood them, solved them, and recorded the outcome.

The Medium writer must never define the problem from the AI service or crawler perspective.  
The writing perspective is always:

> A learner who studied the lecture, article, lab, or exercise and recorded difficult, complex, or practically important learning points as a problem-solving portfolio article.

---

# Completion Score Rule

Each item is scored from 1 to 10.

| Score | Meaning |
|---:|---|
| 1 | Not started |
| 2 | Rough idea only |
| 3 | Skeleton created |
| 4 | Basic implementation started |
| 5 | Works for one simple case |
| 6 | Works for multiple controlled cases |
| 7 | Handles common failure cases |
| 8 | Stable enough for internal testing |
| 9 | Stable enough for user testing |
| 10 | Production-ready for MVP |

---

# Current Overall Status

| Area | Score / 10 | Status |
|---|---:|---|
| Stable image/sample article generation | 8 | Example 1, 2, 3 tested. Existing image-based workflow should be protected. |
| AI Skills Navigator URL collection | 7 | Tested in separate branch, but not merged into main. Needs isolation from universal collector. |
| Universal URL collector branch setup | 6 | `feature/universal-learning-source-collector-v0` branch exists and tracks origin. |
| Universal collector implementation | 4 | v0 skeleton works for YouTube and Agent Academy smoke tests. |
| Medium writer role definition | 8 | Learner-perspective problem-solving template is defined, but must be protected in prompts/code. |
| Visible trace / user-verifiable collection | 4 | trace.jsonl and collection files are generated. Browser-visible mode still needs UI integration. |
| Quality gate before article generation | 4 | pass/partial/fail and can_generate_article are working in v0 collector output. |

---

# 1. Product Goal Definition

## Goal

Build a collector that can accept many kinds of learning URLs and collect enough evidence to generate a problem-solving Medium article.

## Checklist

| Item | Score / 10 | Done |
|---|---:|---|
| Define the app as a learning capture and portfolio-writing tool | 8 | [x] |
| Define the writer as a learner, not an AI system developer | 8 | [x] |
| Define URL collection as evidence gathering, not final writing | 7 | [x] |
| Define source collection before Medium article generation | 7 | [x] |
| Define partial/fail states instead of forcing article generation | 6 | [ ] |
| Define MVP scope similar to lecture-summary tools such as Thetawave-level collection, plus our Medium template | 7 | [x] |

---

# 2. Branch and Safety Rules

## Goal

Protect stable sample/image generation and isolate URL collector experiments.

## Checklist

| Item | Score / 10 | Done |
|---|---:|---|
| Keep stable image/sample generation separate from universal collector | 8 | [x] |
| Preserve `feature/url-deep-source-collector-v2` as AI Skills-oriented branch | 7 | [x] |
| Use `feature/universal-learning-source-collector-v0` for universal collector work | 7 | [x] |
| Do not merge universal collector into main before testing | 8 | [x] |
| Do not deploy to Hugging Face before collection quality passes | 8 | [x] |
| Do not keep patch/debug zip files in Git | 6 | [ ] |
| Keep `.env`, `.venv`, source runs, captures, and large files out of Git | 6 | [ ] |

---

# 3. URL Classification Design

## Goal

Classify URLs by collection problem, not only by website name.

## Classification Axes

### Content Shape

- text_only
- video_only
- text_video_mixed
- lab_or_exercise
- code_heavy
- image_or_slide_heavy

### Navigation Shape

- single_page
- toc_based
- sidebar_course
- next_prev_sequence
- child_pages
- dynamic_app

### Access Level

- public
- login_required
- enrollment_required
- partially_visible
- blocked_or_drm

### Evidence Targets

- main_text
- transcript
- headings
- table_of_contents
- child_links
- images
- code_blocks
- tables
- lab_steps
- exercise_steps
- external_references

## Checklist

| Item | Score / 10 | Done |
|---|---:|---|
| Build domain hint detector | 1 | [ ] |
| Build content-shape detector | 1 | [ ] |
| Build navigation-shape detector | 1 | [ ] |
| Build access-level detector | 1 | [ ] |
| Build evidence-target planner | 1 | [ ] |
| Return a structured collection plan before crawling | 1 | [ ] |

---

# 4. Universal Collector Architecture

## Goal

Keep `app/main.py` thin and move collection logic into tools.

## Target Structure

```text
tools/
  universal_learning_collector.py
  run_universal_collector_smoke.py

tools/collector_core/
  detector.py
  schema.py
  trace.py
  quality_gate.py
  source_pack_writer.py
  evidence_ranker.py

tools/extractors/
  youtube.py
  agent_academy.py
  wikidocs.py
  oopy.py
  ai_skills.py
  protected_course.py
  generic_web.py


## 2026-06-13

| Area | Score / 10 | Notes |
|---|---:|---|
| YouTube extractor | 6 | `d2X38zE7VsU` transcript collected successfully: 2,222 transcript segments, 42,540 text chars, quality pass. |
| Agent Academy extractor | 5 | `microsoft.github.io/agent-academy` collected 7 pages, 62,693 text chars, 140 images, 29 code blocks, quality pass. Lesson-level node splitting still needs improvement. |
| Universal collector skeleton | 4 | source_graph, collection_report, source_pack, and trace outputs are generated. |
| Quality gate | 4 | pass/partial/fail and can_generate_article are working at collector level. |
| Medium writer integration | 2 | Not connected yet by design. |


## 2026-06-13 Oopy TOC-only Smoke Test

| Area | Score / 10 | Notes |
|---|---:|---|
| Oopy extractor | 6 | Root page TOC detected and collected correctly: 18 TOC candidates, 18 collected, 0 missing, 100% TOC coverage, 0 extra pages outside TOC. |
| Oopy quality gate | 5 | `toc_candidates`, `toc_collected`, `toc_missing`, `toc_coverage`, and `extra_collected_outside_toc` are now included in quality output. |
| Oopy limitation | 4 | Image counting currently reflects limited/root evidence and should later aggregate images across TOC pages if needed. |


## 2026-06-13 Oopy TOC-only Smoke Test

| Area | Score / 10 | Notes |
|---|---:|---|
| Oopy extractor | 6 | Root page TOC detected and collected correctly: 18 TOC candidates, 18 collected, 0 missing, 100% TOC coverage, 0 extra pages outside TOC. |
| Oopy quality gate | 5 | `toc_candidates`, `toc_collected`, `toc_missing`, `toc_coverage`, and `extra_collected_outside_toc` are now included in quality output. |
| Oopy limitation | 4 | Image counting currently reflects limited/root evidence and should later aggregate images across TOC pages if needed. |


## 2026-06-13 Protected Course Decision

| Area | Score / 10 | Notes |
|---|---:|---|
| Inflearn/Udemy protected lecture pages | 3 | v0 intentionally treats protected lecture pages as partial/unsupported unless visible transcript or public curriculum content is available. |
| Protected course safety behavior | 5 | The collector marks protected lecture pages as partial, records missing transcript/login/enrollment/DRM limitations, and disables article generation. |
| Product decision | 7 | The app should focus on public learning sources and user-provided study evidence, not bypassing login-only or DRM-protected course content. |
