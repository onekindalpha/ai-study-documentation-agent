# Study Capture Copilot

Lecture capture-to-technical blog generation service that turns screenshots, course URLs, lab instructions, and rough notes into problem-solving Medium-style drafts.

## Demo

- Live Demo: (https://huggingface.co/spaces/onekindalpha/study-capture-copilot)

## What It Does

Study Capture Copilot helps learners convert fragmented study evidence into structured technical writing.

It supports lecture screenshots, course URLs, lab instruction URLs, rough notes, and optional Q&A records.

## Key Features

- Screenshot-based learning reconstruction
- URL-assisted technical article generation
- Image-only fallback for lecture captures
- Practical problem selection for portfolio writing
- Medium-style technical blog draft generation
- Markdown copy support
- Generated draft reset
- Input field reset

## Architecture

```mermaid
flowchart TB
    IMG["Lecture Screenshots"] --> CAP["Capture Analyzer"]
    URL["Course / Video / Lab URLs"] --> SRC["Source Pack Builder"]
    NOTE["Rough Notes"] --> SRC
    QA["Optional Q&A Records"] --> SRC

    CAP --> TOPIC["Topic Detector"]
    SRC --> TOPIC

    TOPIC --> PROBLEM["Practical Problem Selector"]
    PROBLEM --> PLAN["Article Planner"]
    PLAN --> GEN["Medium-style Markdown Generator"]

    GROQ["Groq Fast Provider"] --> GEN
    DEEP["Optional Deep Provider"] --> GEN

    GEN --> VOICE["Blog Voice Filter"]
    VOICE --> UI["Web UI"]
    UI --> COPY["Copy Markdown"]
    UI --> RESET1["Reset Draft"]
    UI --> RESET2["Reset Inputs"]
