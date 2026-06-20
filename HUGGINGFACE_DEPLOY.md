# Hugging Face Spaces 배포

이 프로젝트는 Hugging Face의 **Docker Space**로 배포합니다.

## 1. Space 생성

1. Hugging Face에서 **New Space**를 선택합니다.
2. Space 이름을 정합니다.
3. SDK는 **Docker**를 선택합니다.
4. 초기 테스트는 **Public**과 무료 CPU 하드웨어를 선택합니다.

## 2. Groq Secret 등록

Space의 **Settings → Variables and secrets → New secret**에서 다음 값을 등록합니다.

```text
Name: GROQ_API_KEY
Value: 본인의 Groq API 키
```

선택 환경변수는 필요할 때 Variables에 등록합니다.

```text
GROQ_MODEL=llama-3.1-8b-instant
GROQ_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
GROQ_VISION_CHUNK_SIZE=3
GROQ_VISION_MAX_TOKENS=1200
```

`GROQ_API_KEY`를 Git 저장소, README, Dockerfile에 직접 작성하면 안 됩니다.

## 3. 프로젝트 업로드

Space 생성 후 표시되는 Git 주소를 사용합니다.

```bash
git remote add space https://huggingface.co/spaces/HF_USERNAME/SPACE_NAME
git push space restore/image-evidence-writer:main
```

Hugging Face 계정 비밀번호 대신 Write 권한이 있는 Access Token을 사용합니다.

## 4. 공개 링크 전달

빌드가 완료되면 아래 주소를 초기 사용자에게 전달합니다.

```text
https://huggingface.co/spaces/HF_USERNAME/SPACE_NAME
```

## 무료 베타 주의사항

- 모든 사용자가 하나의 Groq 조직 한도를 공유합니다.
- 무료 Space가 재시작되면 로컬 `data/` 기록은 사라질 수 있습니다.
- 초기에는 소수 사용자에게만 링크를 전달합니다.
- 사용자 이미지와 글을 영구 보관하는 서비스로 사용하지 않습니다.
- 트래픽이 늘면 사용자별 제한, 외부 저장소, 비동기 작업 큐를 추가해야 합니다.
