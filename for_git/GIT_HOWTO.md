# Git으로 팀원과 공유하기

## 처음 올리는 경우 (GitHub 새 저장소)

### 1단계: GitHub에서 저장소 생성
1. https://github.com → "New repository"
2. Repository name: `TransFG-CUB200-Study` (또는 원하는 이름)
3. Public / Private 선택
4. **README 체크 해제** (이미 있으므로)
5. "Create repository" 클릭

### 2단계: 로컬에서 git 초기화 및 푸시

```bash
cd /Users/macminim4/Aiffel03/Project01/TransFG

# git 초기화
git init

# 원격 저장소 연결 (GitHub URL로 교체)
git remote add origin https://github.com/<your-username>/TransFG-CUB200-Study.git

# 현재 상태 확인 (.gitignore 적용 확인)
git status

# 파일 추가 (대용량 파일은 .gitignore로 자동 제외)
git add .

# 커밋
git commit -m "feat: TransFG CUB-200-2011 스터디 프로젝트 초기 커밋"

# 푸시
git branch -M main
git push -u origin main
```

---

## git status로 확인할 것들

아래 파일들이 **추적되지 않음(untracked)** 상태여야 합니다:
```
CUB_200_2011/          ← 1.2GB 데이터셋 (제외됨)
pretrained/             ← 394MB 가중치 (제외됨)
TransFG_venv/           ← 가상환경 (제외됨)
output/                 ← 학습 체크포인트 (제외됨)
logs/                   ← 학습 로그 (제외됨)
```

이것들이 `git add .`에 포함되지 않는지 반드시 확인하세요.

---

## 이후 변경사항 올리기

```bash
# 변경된 파일 확인
git status

# 특정 파일만 추가
git add TransFG_CUB200.ipynb
git add for_git/images/새_이미지.png

# 또는 전체 추가 (gitignore 적용됨)
git add .

# 커밋
git commit -m "docs: 학습 결과 이미지 및 설명 추가"

# 푸시
git push
```

---

## Apple Notes 이미지를 git에 추가하는 방법

Apple Notes에 붙여넣은 이미지는 iCloud에 저장되어 로컬 경로가 고정되지 않습니다.
아래 방법으로 이미지를 추출해서 `for_git/images/` 에 저장하세요:

### 방법 A: Apple Notes에서 직접 저장
1. Apple Notes 앱 열기
2. 이미지가 있는 노트 열기
3. 이미지 위에서 **우클릭 → "이미지를 데스크탑에 저장"**
4. 저장된 이미지를 `for_git/images/` 폴더로 이동:
   ```bash
   mv ~/Desktop/Pasted\ Graphic.png \
     /Users/macminim4/Aiffel03/Project01/TransFG/for_git/images/transfg_result.png
   ```
5. git에 추가:
   ```bash
   git add for_git/images/transfg_result.png
   git commit -m "docs: TransFG 실험 결과 이미지 추가"
   git push
   ```

### 방법 B: 스크린샷으로 캡처
```
Cmd + Shift + 4  →  원하는 영역 드래그  →  ~/Desktop에 저장
```
저장 후 `for_git/images/` 로 이동 후 git add.

---

## 팀원이 클론하는 방법

```bash
git clone https://github.com/<your-username>/TransFG-CUB200-Study.git
cd TransFG-CUB200-Study

# 환경 설정은 for_git/SETUP_GUIDE.md 참조
# 데이터셋, 가중치는 별도 다운로드 필요 (SETUP_GUIDE.md 참조)
```

---

## 주의사항

| 하면 안 됨 | 이유 |
|-----------|------|
| `git add CUB_200_2011/` | 1.2GB → 푸시 불가 |
| `git add pretrained/` | 394MB → GitHub 100MB 제한 초과 |
| `git add TransFG_venv/` | 가상환경은 각자 설치 |
| `git add output/*.bin` | 체크포인트 대용량 |

> GitHub 파일 크기 제한: 단일 파일 100MB, 권장 50MB 이하
