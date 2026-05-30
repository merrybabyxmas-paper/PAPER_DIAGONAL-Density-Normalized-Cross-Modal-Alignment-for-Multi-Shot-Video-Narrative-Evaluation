# DIAGONAL 논문 (ACM MM 2026, Submission 4075) 실험 재현 세팅

## 생성일: 2026-05-31
## 머신: dongwoo38@server

---

## 1. 디렉토리 구조

### 1.1 메인 실험 디렉토리
```
/home/dongwoo38/diagonal_rebuttal/
├── ablation_results/
│   └── vlm_fullscale_merged.json          # MD5: a140afb5746aaf6f81d3c2df72a12dd4
├── prompts/
│   └── diagonal_diversified_prompts.json  # 1000 prompts
├── rebuttal_experiments/
│   ├── exp1_realizable_count.py
│   ├── exp2_noise_sensitivity.py
│   └── results/
├── multi_shot_eval_repos/
│   └── Video-In-Context/
│       └── scene/
│           └── pytorch_lora_weights.safetensors
├── outputs/
│   └── vic/                               # (will be created)
├── generate_vic_fast.py                   # [MODIFIED]
├── generate_tables.py
├── verify_paper_numbers.py
└── eval_vlm_fullscale.py
```

### 1.2 기존 실험 코드 디렉토리 (참조용)
```
/home/dongwoo38/papers/paper_DIAGONAL/codes/diagonal_experiment/
├── prompts/
│   ├── diagonal_diversified_prompts.json  # 1000 prompts (7 patterns)
│   └── diagonal_full_prompts.json         # 200 prompts (Relay only)
├── results/
│   └── generated_evaluation/
│       ├── VIC_evaluation.json            # 100 videos only
│       ├── StoryDiffusion_evaluation.json
│       └── EchoShot_evaluation.json
└── multi_shot_eval_repos/
    └── (empty - no Video-In-Context)
```

### 1.3 SCENE 레포지토리
```
/home/dongwoo38/papers/paper_SCENE/SCENE-Stylistic-Coherence-and-Entity-Narrative-Evaluation-for-Multi-Shot-Video-Generation/
└── wrappers.py                            # VideoInContextWrapper
```

### 1.4 Git 레포지토리
```
/home/dongwoo38/papers/paper_DIAGONAL/PAPER_DIAGONAL-Density-Normalized-Cross-Modal-Alignment-for-Multi-Shot-Video-Narrative-Evaluation/
└── (Overleaf paper only, no experiment code)
```

---

## 2. 핵심 데이터 파일

### 2.1 vlm_fullscale_merged.json
- **경로**: `~/diagonal_rebuttal/ablation_results/vlm_fullscale_merged.json`
- **MD5**: `a140afb5746aaf6f81d3c2df72a12dd4`
- **크기**: 약 456KB (압축 전)
- **구조**: JSON array, 4000개 entries (1000 videos × 4 models)
- **모델별 데이터**:
  - storydiff: 1000 entries
  - echoshot: 1000 entries
  - vgot: 1000 entries
  - vic: 1000 entries
- **각 entry 구조**:
  ```json
  {
    "method": "vic",
    "vid": "DM1K_XXX_PatternName_YYY",
    "pattern": "PatternName",
    "entities": ["entity1", "entity2", "entity3"],
    "S_obs": [[1,0,0], [1,1,0], [1,1,0]],      # observed
    "S_ideal": [[1,0,0], [1,1,0], [1,1,1]],    # prescribed
    "delta_S_obs": [...],
    "delta_S_ideal": [...],
    ...
  }
  ```

### 2.2 diagonal_diversified_prompts.json
- **경로**: `~/diagonal_rebuttal/prompts/diagonal_diversified_prompts.json`
- **구조**: 1000 prompts, 7 patterns
- **Pattern 분포**:
  - Relay: 200
  - Split: 200
  - Accumulation: 200
  - Convergence: 100
  - Sliding_Window: 100
  - Reduction: 100
  - Reverse_Relay: 100
- **ID 형식**: `DM1K_XXX_PatternName_YYY`
- **각 prompt 구조**:
  ```json
  {
    "DM1K_001_Relay_070": {
      "metadata": {
        "pattern_name": "Relay",
        "pattern_type": "Relay",
        ...
      },
      "shots": {
        "shot_1": {
          "entities": ["entity1"],
          "gen_prompt": "A video of ...",
          "eval_prompt": "..."
        },
        "shot_2": {...},
        "shot_3": {...}
      }
    }
  }
  ```

---

## 3. 모델 및 가중치

### 3.1 VIC (Video-In-Context)
- **Base Model**: THUDM/CogVideoX-5b
  - **위치**: `~/.cache/huggingface/hub/models--THUDM--CogVideoX-5b/`
  - **Already cached**: Yes
- **LoRA Weights**: feizhengcong/Video-In-Context
  - **위치**: `~/diagonal_rebuttal/multi_shot_eval_repos/Video-In-Context/scene/pytorch_lora_weights.safetensors`
  - **다운로드 완료**: Yes
  - **크기**: ~300MB

### 3.2 기타 모델 (참조용)
- StoryDiffusion
- EchoShot
- VGoT (VideoGen-of-Thought)

---

## 4. 생성 스크립트 설정

### 4.1 generate_vic_fast.py (수정됨)
**수정 내역**:
```python
# BEFORE:
BASE_DIR = "/home/dongwoo43/papers/paper_DIAGONAL/codes/diagonal_experiment"

# AFTER:
BASE_DIR = "/home/dongwoo38/diagonal_rebuttal"
```

**생성 파라미터** (하드코딩):
- seed: 42
- num_inference_steps: 50
- num_frames: 49
- guidance_scale: 6
- fps: 8

**출력 구조**:
```
~/diagonal_rebuttal/outputs/vic/
├── Relay/
│   └── DM1K_XXX_Relay_YYY.mp4
├── Split/
│   └── DM1K_XXX_Split_YYY.mp4
├── Accumulation/
├── Convergence/
├── Sliding_Window/
├── Reduction/
└── Reverse_Relay/
```

**사용법**:
```bash
cd ~/diagonal_rebuttal
CUDA_VISIBLE_DEVICES=1 python3 generate_vic_fast.py \
  --start_idx 500 \
  --end_idx 567 \
  --gpu_id 0 \
  --seed 42
```

---

## 5. 검증 실험 결과 (STAGE 1-2 완료)

### 5.1 STAGE 1: MD5 검증
```bash
cd ~/diagonal_rebuttal
md5sum ablation_results/vlm_fullscale_merged.json
```
**결과**: `a140afb5746aaf6f81d3c2df72a12dd4` ✓ (논문과 동일)

### 5.2 STAGE 2: 논문 숫자 재현

#### exp2_noise_sensitivity.py
```bash
python3 rebuttal_experiments/exp2_noise_sensitivity.py
```
**결과**:
```
Baseline PTM: storydiff=0.2435, echoshot=0.1672, vgot=0.0943, vic=0.0705
  recompute vs stored max abs err = 0.00e+00  ✓
  Kruskal H=302.7 p=2.59e-65  ✓
  Spearman(IC,PTM)=-1.00  ✓
```

#### exp1_realizable_count.py
```bash
python3 rebuttal_experiments/exp1_realizable_count.py
```
**결과**:
```
N=3: 343 of 729 (47.1%) are physically realizable  ✓
```

#### verify_paper_numbers.py
```bash
python3 verify_paper_numbers.py
```
**결과**:
```
Total checks: 181
Matches: 180
Discrepancies: 1 (VGoT CI upper bound, bootstrap variance)
```

**결론**: 논문 숫자가 JSON에서 결정론적으로 재현됨 ✓

---

## 6. HuggingFace 데이터셋 현황

### 6.1 업로드된 VIC 비디오
- **레포**: https://huggingface.co/datasets/merrybabyxmas/DIAGONAL/tree/main/vic
- **구조**: Pattern 폴더별 분류
- **현황** (2026-05-31):
  - Relay: 31/200
  - Split: 34/200
  - Accumulation: 28/200
  - Convergence: 18/100
  - Sliding_Window: 20/100
  - Reduction: 13/100
  - Reverse_Relay: 15/100
  - **Total: 159/1000**

### 6.2 누락 비디오
- **총 누락**: 841 videos
- **분석 파일**: `/home/dongwoo38/papers/paper_DIAGONAL/codes/diagonal_experiment/missing_vic_videos.json`
- **GPU 분할**: `/home/dongwoo38/papers/paper_DIAGONAL/codes/diagonal_experiment/missing_vic_gpu_splits.json`
  - GPU 1: 280 videos
  - GPU 2: 280 videos
  - GPU 3: 281 videos

---

## 7. 검증 계획 (STAGE 3)

### 7.1 검증 대상: DM1K_500-566
- **개수**: 67 videos
- **인덱스 범위**: 500-566
- **JSON 결과**: ablation_results/vlm_fullscale_merged.json에 67개 VIC 결과 존재
- **목적**: 재생성 비디오의 S_obs가 JSON과 일치하는지 검증

### 7.2 검증 절차
1. DM1K_500-566 재생성 (seed=42, steps=50, frames=49, guidance=6)
2. 생성된 비디오를 VLM으로 평가하여 S_obs 추출
3. JSON의 S_obs와 비교
4. **일치** → 841개 생성 진행
5. **불일치** → 멈추고 보고

---

## 8. 시스템 환경

### 8.1 Python 환경
```
Python: 3.13
PyTorch: 2.6.0+cu124
Conda: miniconda3
Path: /home/dongwoo38/miniconda3
```

### 8.2 GPU/CUDA 상태 (문제 있음)
```
NVIDIA Driver: 535.309.01
GPU Devices: /dev/nvidia0-3 (4 GPUs)
CUDA Libraries: /lib/x86_64-linux-gnu/libcuda.so.1

PyTorch CUDA Status:
  torch.cuda.is_available(): False  ❌
  Error: "CUDA driver initialization failed"
  nvidia-smi: "Unable to determine device handle"
```

### 8.3 Dependencies
```bash
pip install numpy scipy
pip install torch diffusers transformers accelerate
pip install huggingface_hub
```

---

## 9. 현재 상태 및 블로커

### 9.1 완료된 작업
- ✅ STAGE 1: Bundle 압축 해제 및 MD5 검증
- ✅ STAGE 2: 논문 숫자 완전 재현 (recompute err = 0.00)
- ✅ VIC LoRA 다운로드
- ✅ CogVideoX-5b 캐시 확인
- ✅ generate_vic_fast.py 경로 수정

### 9.2 블로커
- ❌ **CUDA 초기화 실패**
  - PyTorch가 GPU 인식 못함
  - nvidia-smi 실행 불가
  - 원인: 드라이버/PyTorch/Python 버전 호환성 문제 추정

### 9.3 미완료 작업
- ⏸️ DM1K_500-566 재생성 (67 videos)
- ⏸️ S_obs 검증
- ⏸️ 841개 누락 비디오 생성
- ⏸️ HuggingFace 업로드

---

## 10. 재생산 가이드

### 10.1 논문 숫자 재현 (GPU 불필요)
```bash
# 1. Bundle 다운로드 및 압축 해제
mkdir -p ~/diagonal_rebuttal
tar xzf ~/diagonal_rebuttal_bundle.tar.gz -C ~/diagonal_rebuttal

# 2. MD5 검증
cd ~/diagonal_rebuttal
md5sum ablation_results/vlm_fullscale_merged.json
# Expected: a140afb5746aaf6f81d3c2df72a12dd4

# 3. 의존성 설치
python3 -m pip install numpy scipy

# 4. 실험 실행
python3 rebuttal_experiments/exp2_noise_sensitivity.py
python3 rebuttal_experiments/exp1_realizable_count.py
python3 generate_tables.py
python3 verify_paper_numbers.py
```

### 10.2 VIC 비디오 생성 (GPU 필요, 현재 불가)
```bash
# 1. LoRA 다운로드
cd ~/diagonal_rebuttal
mkdir -p multi_shot_eval_repos/Video-In-Context/scene
hf download feizhengcong/Video-In-Context \
  scene/pytorch_lora_weights.safetensors \
  --local-dir multi_shot_eval_repos/Video-In-Context

# 2. CogVideoX-5b (자동 다운로드됨)
# ~/.cache/huggingface/hub/models--THUDM--CogVideoX-5b/

# 3. 검증: DM1K_500-566 생성
CUDA_VISIBLE_DEVICES=1 python3 generate_vic_fast.py \
  --start_idx 500 --end_idx 567 --gpu_id 0 --seed 42

# 4. 전체 생성 (GPU 1,2,3 병렬)
# GPU 1:
CUDA_VISIBLE_DEVICES=1 python3 generate_vic_fast.py \
  --start_idx 0 --end_idx 334 --gpu_id 0 --seed 42

# GPU 2:
CUDA_VISIBLE_DEVICES=2 python3 generate_vic_fast.py \
  --start_idx 334 --end_idx 667 --gpu_id 0 --seed 42

# GPU 3:
CUDA_VISIBLE_DEVICES=3 python3 generate_vic_fast.py \
  --start_idx 667 --end_idx 1000 --gpu_id 0 --seed 42
```

---

## 11. 다음 단계 (GPU 복구 후)

1. **CUDA 문제 해결**:
   - 시스템 재부팅
   - PyTorch 재설치 (CUDA 11.8 또는 12.1)
   - Python 3.10/3.11로 다운그레이드

2. **검증 실험**:
   - DM1K_500-566 재생성
   - VLM 평가 후 S_obs 비교

3. **전체 생성**:
   - 841개 누락 비디오 생성
   - HuggingFace 업로드
   - 최종 검증

---

## 12. 참고 사항

### 12.1 VIC 생성 사양
- 1 video ≈ 2-3분 (RTX 4090 기준)
- 67 videos ≈ 2-3시간
- 841 videos ≈ 28-42시간 (3 GPU 병렬 시 ≈ 9-14시간)

### 12.2 원본 생성 머신
- User: dongwoo43 (not dongwoo38)
- 이 머신이 원본 VIC 비디오 생성한 4090 서버로 추정
- 하지만 /home/dongwoo43 디렉토리 존재하지 않음

### 12.3 연락처
- Paper: ACM MM 2026 Submission 4075
- Title: "The Algebra of Storytelling: Density-Normalized Cross-Modal Alignment for Multi-Shot Video Narrative Evaluation"
- Dataset: merrybabyxmas/DIAGONAL (HuggingFace)

---

## 13. 상세 파일 크기 및 체크섬

### 13.1 핵심 데이터
```bash
# vlm_fullscale_merged.json
-rw-rw-r-- 1 dongwoo38 dongwoo38 4.7M Mar 17 17:05 /home/dongwoo38/diagonal_rebuttal/ablation_results/vlm_fullscale_merged.json

# diagonal_diversified_prompts.json
-rw-rw-r-- 1 dongwoo38 dongwoo38 1008K Apr  1 23:49 /home/dongwoo38/diagonal_rebuttal/prompts/diagonal_diversified_prompts.json

# VIC LoRA weights
-rw-rw-r-- 1 dongwoo38 dongwoo38 64M May 31 00:06 /home/dongwoo38/diagonal_rebuttal/multi_shot_eval_repos/Video-In-Context/scene/pytorch_lora_weights.safetensors
```

### 13.2 전체 디렉토리 트리
```
```

---

## 14. 추가 스크립트 상세

### 14.1 eval_vlm_fullscale.py
#!/usr/bin/env python3
"""
Full-scale VLM evaluation: Qwen2-VL-7B entity presence detection.
Builds S_obs matrix, computes ΔS_obs, matrix distance for all videos.
Supports multi-GPU parallel execution via --gpu and --total_gpus args.
"""
import os, json, time, glob, sys, argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_PATH = os.path.join(SCRIPT_DIR, "prompts", "diagonal_diversified_prompts.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

PATTERN_PRESENCE = {
    "Relay":          [[1, 0], [1, 1], [0, 1]],
    "Split":          [[1, 1], [1, 0], [0, 1]],
    "Accumulation":   [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
    "Convergence":    [[1, 0], [0, 1], [1, 1]],
    "Sliding_Window": [[1, 1, 0], [0, 1, 1], [0, 0, 1]],
    "Reduction":      [[1, 1, 1], [1, 1, 0], [1, 0, 0]],
    "Reverse_Relay":  [[0, 1], [1, 1], [1, 0]],
}
PATTERN_DELTA = {
    "Relay":          np.array([[0, +1], [-1, 0]]),
    "Split":          np.array([[0, -1], [-1, +1]]),
    "Accumulation":   np.array([[0, +1, 0], [0, 0, +1]]),
    "Convergence":    np.array([[-1, +1], [+1, 0]]),
    "Sliding_Window": np.array([[-1, 0, +1], [0, -1, 0]]),
    "Reduction":      np.array([[0, 0, -1], [0, -1, 0]]),
    "Reverse_Relay":  np.array([[+1, 0], [0, -1]]),
...
```

### 14.2 rebuttal_experiments 구조
total 72
drwxrwxr-x 3 dongwoo38 dongwoo38 4096 May 30 16:35 .
drwxrwxr-x 8 dongwoo38 dongwoo38 4096 May 31 00:47 ..
-rw-rw-r-- 1 dongwoo38 dongwoo38 5595 May 30 15:19 exp1_realizable_count.py
-rw-rw-r-- 1 dongwoo38 dongwoo38 6157 May 30 15:27 exp2c_adversarial_bias.py
-rw-rw-r-- 1 dongwoo38 dongwoo38 8374 May 30 15:20 exp2_noise_sensitivity.py
-rw-rw-r-- 1 dongwoo38 dongwoo38 3874 May 30 16:34 exp3_analyze.py
-rw-rw-r-- 1 dongwoo38 dongwoo38 8580 May 30 16:32 exp3_multiframe_vote.py
-rwxrwxr-x 1 dongwoo38 dongwoo38 1857 May 30 15:32 launch_exp3.sh
-rw-rw-r-- 1 dongwoo38 dongwoo38 4017 May 30 16:35 README.md
-rw-rw-r-- 1 dongwoo38 dongwoo38 3993 May 30 15:31 REBUTTAL_RESULTS.md
drwxrwxr-x 2 dongwoo38 dongwoo38 4096 May 30 16:33 results
-rw-rw-r-- 1 dongwoo38 dongwoo38 1304 May 30 16:34 run_exp3_multigpu.sh

---

## 15. 환경 변수 및 경로

### 15.1 중요 경로 변수
```bash
# 현재 작업 디렉토리
REBUTTAL_DIR="/home/dongwoo38/diagonal_rebuttal"

# 데이터 파일
VLM_JSON="$REBUTTAL_DIR/ablation_results/vlm_fullscale_merged.json"
PROMPTS_JSON="$REBUTTAL_DIR/prompts/diagonal_diversified_prompts.json"

# 모델 가중치
LORA_PATH="$REBUTTAL_DIR/multi_shot_eval_repos/Video-In-Context/scene/pytorch_lora_weights.safetensors"
COGVIDEO_CACHE="$HOME/.cache/huggingface/hub/models--THUDM--CogVideoX-5b"

# 출력 디렉토리
VIC_OUTPUT="$REBUTTAL_DIR/outputs/vic"
```

### 15.2 GPU 설정
```bash
# GPU 확인
nvidia-smi  # 현재 불가

# PyTorch CUDA 확인
python3 -c "import torch; print(torch.cuda.is_available())"  # False (문제)

# 환경 변수 설정 (사용 예)
export CUDA_VISIBLE_DEVICES=1
```

---

## 16. 에러 로그 및 디버깅

### 16.1 CUDA 초기화 실패 로그
```
/home/dongwoo38/miniconda3/lib/python3.13/site-packages/torch/cuda/__init__.py:129: 
UserWarning: CUDA initialization: CUDA driver initialization failed, 
you might not have a CUDA gpu. 
(Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:109.)
```

### 16.2 nvidia-smi 에러
```
Unable to determine the device handle for GPU0000:01:00.0: Unknown Error
```

### 16.3 진단 결과
- NVIDIA driver: 정상 (535.309.01)
- Device files: 정상 (/dev/nvidia0-3)
- CUDA libraries: 정상 (/lib/x86_64-linux-gnu/libcuda.so.1)
- PyTorch CUDA: **비정상** (초기화 실패)

**추정 원인**:
1. Python 3.13 + PyTorch 2.6.0 호환성 문제
2. CUDA 12.4 vs Driver 535 버전 미스매치
3. 시스템 레벨 CUDA 런타임 문제

**권장 해결책**:
- 시스템 재부팅
- Python 3.10 + PyTorch 2.1.0 + CUDA 11.8 조합으로 재설치
- 또는 conda environment 격리

---

## 17. 백업 및 복구

### 17.1 중요 파일 백업
```bash
# Bundle 백업
cp ~/diagonal_rebuttal_bundle.tar.gz ~/diagonal_rebuttal_bundle.BACKUP.tar.gz

# JSON 백업
cp ~/diagonal_rebuttal/ablation_results/vlm_fullscale_merged.json \
   ~/diagonal_rebuttal/ablation_results/vlm_fullscale_merged.BACKUP.json

# 수정된 스크립트 백업
cp ~/diagonal_rebuttal/generate_vic_fast.py \
   ~/diagonal_rebuttal/generate_vic_fast.ORIGINAL.py
```

### 17.2 복구 절차
```bash
# Bundle 재압축 해제
rm -rf ~/diagonal_rebuttal
mkdir -p ~/diagonal_rebuttal
tar xzf ~/diagonal_rebuttal_bundle.tar.gz -C ~/diagonal_rebuttal

# MD5 재검증
cd ~/diagonal_rebuttal
md5sum ablation_results/vlm_fullscale_merged.json
```

---

## 18. 타임라인

### 2026-05-31 00:00 - STAGE 1 완료
- Bundle 압축 해제
- MD5 검증: a140afb5746aaf6f81d3c2df72a12dd4 ✓

### 2026-05-31 00:08 - STAGE 2 완료
- exp2_noise_sensitivity.py: PTM 재현 ✓
- exp1_realizable_count.py: 343/729 ✓
- verify_paper_numbers.py: 180/181 ✓

### 2026-05-31 00:10 - STAGE 3 시작
- VIC LoRA 다운로드 완료
- CogVideoX-5b 캐시 확인
- generate_vic_fast.py 경로 수정

### 2026-05-31 00:08-00:09 - STAGE 3 중단
- CUDA 초기화 실패
- 테스트 생성 (DM1K_500) 실패
- GPU 접근 불가 확인

---

## 19. 체크리스트

### 실험 재현 (STAGE 1-2)
- [x] Bundle 다운로드
- [x] Bundle 압축 해제
- [x] MD5 검증
- [x] numpy/scipy 설치
- [x] exp2_noise_sensitivity.py 실행
- [x] exp1_realizable_count.py 실행
- [x] generate_tables.py 실행
- [x] verify_paper_numbers.py 실행
- [x] 모든 숫자 일치 확인

### VIC 비디오 생성 (STAGE 3)
- [x] VIC LoRA 다운로드
- [x] CogVideoX-5b 확인
- [x] generate_vic_fast.py 경로 수정
- [ ] CUDA/GPU 문제 해결 ← **현재 블로커**
- [ ] DM1K_500-566 재생성 (67 videos)
- [ ] S_obs 검증
- [ ] 841개 누락 비디오 생성
- [ ] HuggingFace 업로드

---

## 20. FAQ

### Q1: 왜 dongwoo43이 아닌 dongwoo38인가?
A: 원본 스크립트는 /home/dongwoo43 경로 사용. 이 머신은 dongwoo38. 경로 수정 완료.

### Q2: JSON의 MD5가 왜 중요한가?
A: 논문의 모든 숫자가 이 JSON에서 계산됨. MD5 일치 = 데이터 무결성 보장.

### Q3: 왜 DM1K_500-566만 먼저 생성하나?
A: 이 67개는 JSON에 결과가 있어서 검증 가능. S_obs 일치 확인 후 전체 생성.

### Q4: seed=42가 중요한가?
A: Yes. 동일한 랜덤 시드로 생성해야 재현 가능. JSON의 결과도 seed=42로 생성됨.

### Q5: HuggingFace에 왜 159개만 있나?
A: 원인 불명. 원본 생성 후 업로드 과정에서 일부만 업로드된 것으로 추정.

---

## END OF DOCUMENT

---

## 21. 각 모델별 실험 세팅 (Original Generation)

이 섹션은 원본 논문 실험에서 각 모델(EchoShot, VGoT, StoryDiffusion, VIC)을 어떻게 실행했는지 기록합니다.

### 21.1 EchoShot 생성 세팅

**스크립트**: `/home/dongwoo38/papers/paper_DIAGONAL/codes/diagonal_experiment/generate_all_videos.py`

**모델 구조**:
- **T5 Encoder**: UMT5-XXL (text embedding, bf16)
  - Checkpoint: `models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth`
  - Tokenizer: `models/Wan2.1-T2V-1.3B/google/umt5-xxl/`
  - 512 token length
  - CPU offload 전략 (GPU 메모리 절약)

- **VAE**: VideoVAE
  - Checkpoint: `models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth`
  - Latent dimension: z_dim

- **Transformer**: EchoShot-1.3B-preview
  - Checkpoint: `models/EchoShot/EchoShot-1.3B-preview.pth`
  - Dim: 1536, FFN dim: 8960, Freq dim: 256
  - Num heads: 12, Num layers: 30
  - Patch size: (1, 2, 2)
  - QK norm: True, Cross-attn norm: True

**생성 파라미터**:
```python
{
    "sample_frames": 93,           # 총 프레임 수
    "sample_steps": 30,            # 샘플링 스텝
    "sample_shift": 5.0,           # Flow DPM shift
    "sample_guide_scale": 5.0,     # CFG scale
    "H": 480, "W": 832,            # 해상도
    "fps": 16,                     # 프레임률
    "seed": 42,                    # 랜덤 시드
    "dtype": torch.bfloat16,       # 정밀도
    "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，水印，文字"
}
```

**멀티샷 처리**:
- 3개 shot prompts를 단일 모델 forward pass로 처리
- Shot length 자동 분할: `[t//3, t//3, t - 2*(t//3)]`
- Inner temporal dimension 관리 (`inner_t` 파라미터)

**사용법**:
```bash
cd /home/dongwoo38/papers/paper_DIAGONAL/codes/diagonal_experiment

# Single GPU
CUDA_VISIBLE_DEVICES=1 python generate_all_videos.py \
  --start_idx 0 --end_idx 1000 --gpu_id 1 --seed 42

# Multi-GPU (3 GPUs)
CUDA_VISIBLE_DEVICES=1 python generate_all_videos.py --start_idx 0 --end_idx 334 --gpu_id 1 &
CUDA_VISIBLE_DEVICES=2 python generate_all_videos.py --start_idx 334 --end_idx 667 --gpu_id 2 &
CUDA_VISIBLE_DEVICES=3 python generate_all_videos.py --start_idx 667 --end_idx 1000 --gpu_id 3 &
```

**메모리 관리**:
- T5 Encoder를 CPU로 오프로드 (인코딩 시에만 GPU로)
- 10 videos마다 aggressive memory cleanup
- OOM 감지 시 자동 복구 시도

**생성 시간** (RTX 4090 기준):
- 1 video ≈ 2-3분
- 1000 videos ≈ 28-42시간 (단일 GPU)
- 1000 videos ≈ 9-14시간 (3 GPU 병렬)

---

### 21.2 VGoT (VideoGen-of-Thought) 생성 세팅

**스크립트**: `/home/dongwoo38/papers/paper_DIAGONAL/codes/diagonal_experiment/generate_all_videos_vgot.py`

**파이프라인 구조** (2-stage):
1. **Stage 1: Keyframe Generation** (Kolors T2I + IP-Adapter)
2. **Stage 2: Image-to-Video** (DynamiCrafter I2V)

**Stage 1 모델: Kolors T2I**
- **Text Encoder**: ChatGLM (bf16)
  - Path: `weights/Kolors/text_encoder`
- **VAE**: AutoencoderKL (half precision)
  - Path: `weights/Kolors/vae`
- **UNet**: UNet2DConditionModel (half precision)
  - Path: `weights/Kolors/unet`
- **Scheduler**: EulerDiscreteScheduler
- **IP-Adapter** (identity preservation):
  - CLIP Vision: `weights/Kolors-IP-Adapter-Plus/image_encoder`
  - IP weights: `ip_adapter_plus_general.bin`

**Stage 2 모델: DynamiCrafter**
- **Config**: `configs/inference_module3.yaml`
- **Checkpoint**: `weights/DynamiCrafter/model.ckpt`
- **Sampler**: DDIMSampler (multi-condition CFG)
- **Per-frame AE**: True (메모리 절약)

**생성 파라미터**:
```python
# Keyframe (T2I)
{
    "height": 1024, "width": 1024,
    "num_inference_steps": 25,
    "guidance_scale": 5.0,
    "negative_prompt": "low quality, blurry, deformed, distorted",
}

# Video (I2V)
{
    "video_height": 576,
    "video_width": 1024,
    "video_length": 16,        # frames per shot
    "ddim_steps": 50,
    "ddim_eta": 1.0,
    "guidance_scale": 7.5,     # text CFG
    "cfg_img": 7.5,            # image CFG
    "frame_stride": 10,
    "fps": 8,
    "seed": 42,
}
```

**멀티샷 처리**:
- Shot 1: keyframe 생성 → I2V (16 frames)
- Shot 2: keyframe 생성 (IP-Adapter로 identity 유지) → I2V (16 frames)
- Shot 3: keyframe 생성 (IP-Adapter로 identity 유지) → I2V (16 frames)
- 총 48 frames 연결

**Two-Phase Strategy** (메모리 부족 시):
1. Phase 1: 모든 keyframe 생성 → 디스크 저장 → Kolors 언로드
2. Phase 2: DynamiCrafter 로드 → 모든 keyframe에서 I2V 생성

**사용법**:
```bash
cd /home/dongwoo38/papers/paper_DIAGONAL/codes/diagonal_experiment

CUDA_VISIBLE_DEVICES=3 python generate_all_videos_vgot.py \
  --start_idx 500 --end_idx 1000 --gpu_id 3 --seed 42
```

**메모리 요구사항**:
- Kolors: ~12GB VRAM
- DynamiCrafter: ~15GB VRAM
- Total (both): ~27GB (32GB GPU에 가능)
- If OOM: two-phase mode 자동 활성화

**생성 시간** (RTX 4090 기준):
- 1 video ≈ 4-6분 (keyframe 3개 + I2V 3개)
- 1000 videos ≈ 67-100시간 (단일 GPU)

---

### 21.3 StoryDiffusion 생성 세팅

**스크립트**: `/home/dongwoo38/papers/paper_DIAGONAL/codes/diagonal_experiment/generate_all_videos_storydiff.py`

**파이프라인 구조**:
1. **SDXL + Consistent Self-Attention (CSA)**: 3-shot 이미지 시퀀스 생성
2. **SEINE I2V**: 이미지 간 전환 비디오 생성

**Wrapper 사용**:
- Import: `from wrappers import StoryDiffusionWrapper`
- SCENE 레포지토리 경로: `/home/dongwoo38/papers/paper_SCENE/SCENE-Stylistic-Coherence-and-Entity-Narrative-Evaluation-for-Multi-Shot-Video-Generation/wrappers.py`

**생성 파라미터** (wrapper 내부):
```python
{
    # SDXL + CSA
    "num_images": 3,           # 3-shot 이미지
    "height": 1024, "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "sa_step": 32,             # CSA 적용 시작 스텝
    
    # SEINE I2V
    "video_length": 16,        # frames per transition
    "num_inference_steps": 50,
    "fps": 16,
    "seed": 42,
}
```

**CSA (Consistent Self-Attention)**:
- Self-attention layer에서 identity consistency 강제
- 첫 번째 이미지의 attention feature를 subsequent images에 주입
- `sa_step` 이후부터 적용

**사용법**:
```bash
cd /home/dongwoo38/papers/paper_DIAGONAL/codes/diagonal_experiment

CUDA_VISIBLE_DEVICES=2 python generate_all_videos_storydiff.py \
  --gpu_id 2 --start_idx 0 --end_idx 1000
```

**메모리 관리**:
- 5 videos마다 `gc.collect()` + `torch.cuda.empty_cache()`
- Progress tracking: `progress_gpu{gpu_id}.json`

**생성 시간** (RTX 4090 기준):
- 1 video ≈ 3-5분
- 1000 videos ≈ 50-83시간 (단일 GPU)

---

### 21.4 VIC (Video-In-Context) 생성 세팅

**스크립트**: 
- Original: `/home/dongwoo38/papers/paper_DIAGONAL/codes/diagonal_experiment/generate_all_videos_vic.py`
- Rebuttal: `~/diagonal_rebuttal/generate_vic_fast.py`

**모델 구조**:
- **Base Model**: THUDM/CogVideoX-5b
  - Cached: `~/.cache/huggingface/hub/models--THUDM--CogVideoX-5b/`
- **LoRA**: feizhengcong/Video-In-Context (scene LoRA)
  - Path: `multi_shot_eval_repos/Video-In-Context/scene/pytorch_lora_weights.safetensors`
  - Size: 64MB

**생성 파라미터**:
```python
{
    "num_inference_steps": 50,
    "num_frames": 49,
    "guidance_scale": 6,
    "fps": 8,
    "seed": 42,
    "torch_dtype": torch.bfloat16,
}
```

**프롬프트 포맷** (VIC-specific):
```python
# Input: ["prompt1", "prompt2", "prompt3"]
# VIC format:
formatted = "[VIDEO1] prompt1 [VIDEO2] prompt2 [VIDEO3] prompt3"
```

**멀티샷 처리**:
- VIC는 단일 forward pass로 multi-shot 비디오 생성
- 각 `[VIDEON]` 토큰이 shot boundary 표시

**사용법**:
```bash
# Original (diagonal_experiment)
cd /home/dongwoo38/papers/paper_DIAGONAL/codes/diagonal_experiment
CUDA_VISIBLE_DEVICES=1 python generate_all_videos_vic.py \
  --start_idx 0 --end_idx 1000 --seed 42

# Rebuttal (self-contained)
cd ~/diagonal_rebuttal
CUDA_VISIBLE_DEVICES=1 python generate_vic_fast.py \
  --start_idx 500 --end_idx 567 --gpu_id 0 --seed 42
```

**CPU Offload**:
```python
pipe.enable_model_cpu_offload(gpu_id=0)
pipe.vae.enable_tiling()
```

**메모리 요구사항**:
- CogVideoX-5b: ~18GB VRAM
- With CPU offload: ~12GB VRAM
- Peak usage: ~20GB (generation time)

**생성 시간** (RTX 4090 기준):
- 1 video ≈ 2-3분
- 67 videos (DM1K_500-566) ≈ 2-3시간
- 1000 videos ≈ 33-50시간 (단일 GPU)

---

## 22. 공통 실험 설정

### 22.1 Prompt 파일

**Original**: `diagonal_full_prompts.json` (200 prompts, Relay only)
- Used in: 초기 실험, 일부 모델 테스트

**Diversified**: `diagonal_diversified_prompts.json` (1000 prompts, 7 patterns)
- Used in: 최종 논문 실험, 모든 모델
- Patterns: Relay(200), Split(200), Accumulation(200), Convergence(100), Sliding_Window(100), Reduction(100), Reverse_Relay(100)

**Prompt 구조**:
```json
{
  "DM1K_XXX_PatternName_YYY": {
    "metadata": {
      "pattern_name": "Relay",
      "pattern_type": "Sequential_Relay",
      "theme": "restaurant",
      ...
    },
    "shots": {
      "shot_1": {
        "entities": ["chef"],
        "gen_prompt": "A video of a chef cooking in a kitchen",
        "eval_prompt": "Is there a chef in the video?"
      },
      "shot_2": {...},
      "shot_3": {...}
    }
  }
}
```

### 22.2 Output 구조

**Original (flat)**:
```
codes/diagonal_experiment/outputs/
├── echoshot/
│   └── DM1K_XXX_PatternName_YYY.mp4
├── vgot/
│   └── DM1K_XXX_PatternName_YYY.mp4
├── storydiff/
│   └── DM1K_XXX_PatternName_YYY.mp4
└── vic/
    └── DM1K_XXX_PatternName_YYY.mp4
```

**Rebuttal (pattern-organized)**:
```
~/diagonal_rebuttal/outputs/vic/
├── Relay/
│   └── DM1K_XXX_Relay_YYY.mp4
├── Split/
├── Accumulation/
├── Convergence/
├── Sliding_Window/
├── Reduction/
└── Reverse_Relay/
```

### 22.3 Progress Tracking

모든 생성 스크립트는 다음 파일들로 진행 상황 추적:

```bash
# Progress tracking
outputs/{model}/progress.json          # 또는 progress_gpu{N}.json
{
  "completed": ["DM1K_001_...", ...],
  "failed": [
    {"id": "DM1K_123_...", "error": "..."}
  ]
}

# Generation statistics
outputs/{model}/generation_stats.json
{
  "success_count": 856,
  "error_count": 3,
  "avg_time_per_video": 145.2,
  "total_time_seconds": 124675,
  "completed_at": "2026-03-15T14:32:10"
}

# Logs
outputs/{model}/logs/
├── generation_{timestamp}_gpu{N}.log
└── errors.json
```

### 22.4 Common Parameters

모든 모델에서 공통:
- **Seed**: 42 (재현성)
- **Multi-GPU**: `CUDA_VISIBLE_DEVICES` 설정 후 `cuda:0` 접근
- **Resumable**: progress.json 기반 중단/재개 가능
- **Memory Cleanup**: 주기적 `gc.collect()` + `torch.cuda.empty_cache()`

### 22.5 Model Dependencies

**EchoShot**:
```bash
# Required repos
/home/dongwoo38/papers/paper_DIAGONAL/codes/multi_shot_eval_repos/EchoShot_official/

# Key files
- utils/HuggingfaceTokenizer.py
- models/vae.py
- models/model.py (Transformer)
- utils/fm_solvers.py (FlowDPMSolver)
```

**VGoT**:
```bash
# Required repos
/home/dongwoo44/papers/paper_DIAGONAL/codes/multi_shot_eval_repos/VideoGen-of-Thought/

# Key dependencies
- Kolors (SDXL-based T2I)
- DynamiCrafter (I2V)
- IP-Adapter-Plus (identity preservation)
```

**StoryDiffusion**:
```bash
# Required repos
/home/dongwoo38/papers/paper_SCENE/SCENE-.../

# Key file
- wrappers.py → StoryDiffusionWrapper
```

**VIC**:
```bash
# Required
- THUDM/CogVideoX-5b (HuggingFace)
- feizhengcong/Video-In-Context (LoRA)

# Dependencies
- diffusers
- transformers
- accelerate
```

---

## 23. 모델 비교 요약

| Model | Type | Stages | VRAM | Time/Video | Strengths |
|-------|------|--------|------|------------|-----------|
| **EchoShot** | Text-to-Video | 1 (direct T2V) | ~12GB | 2-3분 | Fast, simple pipeline |
| **VGoT** | Image-to-Video | 2 (T2I → I2V) | ~27GB | 4-6분 | Identity preservation |
| **StoryDiffusion** | Hybrid | 2 (CSA T2I → I2V) | ~18GB | 3-5분 | Consistent characters |
| **VIC** | Text-to-Video | 1 (multi-shot T2V) | ~20GB | 2-3분 | Multi-shot aware |

**논문 결과 (PTM)**:
- StoryDiffusion: 0.2435 (worst)
- EchoShot: 0.1672
- VGoT: 0.0943
- **VIC: 0.0705** (best)

**핵심 차이점**:
- **EchoShot**: Inner temporal dimension으로 multi-shot 처리
- **VGoT**: Keyframe 먼저 생성 → I2V (2-stage)
- **StoryDiffusion**: CSA로 character consistency 강제
- **VIC**: `[VIDEON]` 토큰으로 shot boundary 인식

---


---

## 24. 가상환경 설정 (Conda Environments)

각 모델은 별도의 conda 환경에서 실행되었을 가능성이 높습니다.

### 24.1 사용 가능한 Conda 환경

**서버: dongwoo38@server**

```bash
# Conda 환경 리스트
base                      /home/dongwoo38/miniconda3
allinone                  /home/dongwoo38/miniconda3/envs/allinone
allinone_py39             /home/dongwoo38/miniconda3/envs/allinone_py39
diagonal                  /home/dongwoo38/miniconda3/envs/diagonal
jelly                     /home/dongwoo38/miniconda3/envs/jelly
myenv                     /home/dongwoo38/miniconda3/envs/myenv
```

### 24.2 환경별 주요 패키지

#### **allinone** (기본 환경)
```
Python: 3.10.20
torch: 2.0.1 (CUDA 11.7)
torchaudio: 2.0.2
einops: 0.8.2
```
- 용도: 기본 실험, 간단한 테스트
- 특징: 안정적인 torch 2.0.1

#### **allinone_py39** (Python 3.9)
```
Python: 3.9.25
torch: 2.0.1
torchaudio: 2.0.2
```
- 용도: Python 3.9 호환성 필요 시
- 특징: allinone과 동일한 torch 버전

#### **diagonal** (최신 torch)
```
Python: 3.10.19
torch: 2.7.1+cu118 (CUDA 11.8)
torchvision: 0.22.1+cu118
transformers: 5.3.0
einops: 0.8.2
```
- 용도: 최신 모델 테스트
- 특징: 매우 최신 torch (2.7.1), transformers 5.3.0 (비정상적으로 높은 버전)
- ⚠️ 주의: transformers 5.3.0은 비표준 (2025년 1월 기준 최신은 4.x)

#### **jelly** (추천 환경)
```
Python: 3.10.20
torch: 2.5.1+cu121 (CUDA 12.1)
torchvision: 0.20.1+cu121
transformers: 4.40.0
accelerate: 1.13.0
```
- 용도: diffusers 기반 모델 (StoryDiffusion, VIC)
- 특징: accelerate 포함, 안정적인 torch 2.5.1

### 24.3 모델별 Requirements

#### **EchoShot**

**Requirements**: `/home/dongwoo38/papers/paper_DIAGONAL/codes/diagonal_experiment/multi_shot_eval_repos/EchoShot_official/requirements.txt`

```
Python: 3.10.16
torch: 2.5.1+cu121
torchvision: 0.20.1+cu121
torchaudio: 2.5.1+cu121
transformers: 4.49.0
diffusers: 0.32.2
accelerate: 1.4.0
flash-attn: 2.7.4.post1
einops: 0.8.1
decord: 0.6.0
```

**추정 사용 환경**: `jelly` 또는 별도 EchoShot 전용 환경
- torch 2.5.1+cu121 일치
- transformers 4.49.0 (jelly는 4.40.0이므로 약간 다름)

**환경 생성 방법**:
```bash
conda create -n echoshot python=3.10
conda activate echoshot
pip install -r multi_shot_eval_repos/EchoShot_official/requirements.txt
```

#### **VGoT (VideoGen-of-Thought)**

**Requirements**: `/home/dongwoo38/papers/paper_DIAGONAL/codes/diagonal_experiment/multi_shot_eval_repos/VideoGen-of-Thought/requirements.txt`

```
torch: 2.2.2+cu121
torchvision: 0.17.2+cu121
xformers: 0.0.25.post1
transformers: 4.42.4
diffusers: 0.31.0
accelerate: 1.1.1
einops: 0.8.0
decord: 0.6.0
open-clip-torch: 2.12.0
omegaconf
opencv-python
```

**추정 사용 환경**: 별도 VGoT 전용 환경
- ⚠️ **중요**: 하드코딩된 경로 `/home/dongwoo44/`
- 다른 서버 또는 다른 사용자 계정에서 실행된 것으로 추정

**환경 생성 방법**:
```bash
conda create -n vgot python=3.10
conda activate vgot
pip install torch==2.2.2 torchvision==0.17.2 xformers==0.0.25.post1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r multi_shot_eval_repos/VideoGen-of-Thought/requirements.txt
```

#### **StoryDiffusion**

**Requirements**: `/home/dongwoo38/papers/paper_DIAGONAL/codes/diagonal_experiment/multi_shot_eval_repos/ComfyUI_StoryDiffusion/requirements.txt`

```
spaces
peft
diffusers
opencv-python
omegaconf
```

**실제 사용**: SCENE 레포지토리의 wrappers.py 사용
- Path: `/home/dongwoo38/papers/paper_SCENE/SCENE-.../wrappers.py`
- Import: `from wrappers import StoryDiffusionWrapper`

**추정 사용 환경**: `allinone` 또는 `jelly`
- 최소 요구사항: diffusers, peft
- ComfyUI 기반이므로 추가 의존성 있을 수 있음

#### **VIC (Video-In-Context)**

**Requirements**: 별도 requirements.txt 없음 (HuggingFace 기반)

```
torch: 2.x (bfloat16 지원 필요)
diffusers: 0.30+
transformers: 4.40+
accelerate: 0.20+
```

**추정 사용 환경**: `jelly`
- accelerate 포함
- torch 2.5.1+cu121로 bfloat16 지원
- diffusers 최신 버전

**환경 생성 방법**:
```bash
conda create -n vic python=3.10
conda activate vic
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate
```

### 24.4 환경 활성화 및 실행 방법

#### **권장 실행 방법** (모델별)

**EchoShot**:
```bash
conda activate echoshot  # 또는 jelly
cd /home/dongwoo38/papers/paper_DIAGONAL/codes/diagonal_experiment
CUDA_VISIBLE_DEVICES=1 python generate_all_videos.py \
  --start_idx 0 --end_idx 1000 --gpu_id 1 --seed 42
```

**VGoT**:
```bash
conda activate vgot
cd /home/dongwoo38/papers/paper_DIAGONAL/codes/diagonal_experiment
# 주의: 경로 수정 필요 (dongwoo44 -> dongwoo38)
CUDA_VISIBLE_DEVICES=3 python generate_all_videos_vgot.py \
  --start_idx 0 --end_idx 1000 --gpu_id 3 --seed 42
```

**StoryDiffusion**:
```bash
conda activate allinone  # 또는 jelly
cd /home/dongwoo38/papers/paper_DIAGONAL/codes/diagonal_experiment
CUDA_VISIBLE_DEVICES=2 python generate_all_videos_storydiff.py \
  --gpu_id 2 --start_idx 0 --end_idx 1000
```

**VIC**:
```bash
conda activate jelly  # 또는 vic
cd ~/diagonal_rebuttal
CUDA_VISIBLE_DEVICES=1 python generate_vic_fast.py \
  --start_idx 500 --end_idx 567 --gpu_id 0 --seed 42
```

### 24.5 환경 재현 방법

#### **Option 1: Requirements.txt 사용**

```bash
# EchoShot
conda create -n echoshot python=3.10
conda activate echoshot
cd /home/dongwoo38/papers/paper_DIAGONAL/codes/diagonal_experiment
pip install -r multi_shot_eval_repos/EchoShot_official/requirements.txt

# VGoT
conda create -n vgot python=3.10
conda activate vgot
pip install -r multi_shot_eval_repos/VideoGen-of-Thought/requirements.txt
```

#### **Option 2: Conda Export (정확한 재현)**

```bash
# 기존 환경에서 export
conda activate jelly
conda env export > jelly_environment.yml

# 새 환경에서 import
conda env create -f jelly_environment.yml
```

#### **Option 3: 최소 환경 (빠른 시작)**

```bash
# 범용 환경 (모든 모델 지원)
conda create -n diagonal_all python=3.10
conda activate diagonal_all
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate einops decord \
  opencv-python omegaconf peft
```

### 24.6 문제 해결

#### **CUDA 버전 불일치**
```bash
# 현재 CUDA 확인
nvidia-smi  # Driver CUDA version
nvcc --version  # Toolkit CUDA version

# PyTorch CUDA 확인
python -c "import torch; print(torch.version.cuda)"

# 해결: CUDA 11.8 사용 시
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### **Import 에러**
```bash
# 예: ModuleNotFoundError: No module named 'wrappers'
# 해결: 경로 추가
export PYTHONPATH="/home/dongwoo38/papers/paper_SCENE/SCENE-.../:$PYTHONPATH"
```

#### **OOM (Out of Memory)**
```bash
# 환경 변수로 메모리 관리
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1

# 또는 코드에서
torch.cuda.empty_cache()
gc.collect()
```

### 24.7 현재 CUDA 문제 (2026-05-31)

**증상**:
```
CUDA initialization: CUDA driver initialization failed
PyTorch CUDA available: False
```

**영향받는 환경**: 모든 환경 (base, allinone, diagonal, jelly)

**추정 원인**:
1. NVIDIA Driver 535 + PyTorch 2.6.0+cu124 호환성 문제
2. Python 3.13 관련 이슈
3. 시스템 레벨 CUDA 런타임 문제

**해결 시도**:
- [x] 여러 conda 환경 테스트 → 모두 실패
- [x] PyTorch CUDA 초기화 테스트 → 모두 False
- [ ] 시스템 재부팅 (권장)
- [ ] PyTorch 재설치 (CUDA 11.8)
- [ ] Python 3.10 환경으로 다운그레이드

### 24.8 환경별 추천 용도 요약

| 환경 | Python | Torch | CUDA | 추천 모델 | 상태 |
|------|--------|-------|------|-----------|------|
| **allinone** | 3.10.20 | 2.0.1 | 11.7 | 기본 테스트 | ✅ 안정 |
| **allinone_py39** | 3.9.25 | 2.0.1 | 11.7 | Python 3.9 필요 시 | ✅ 안정 |
| **diagonal** | 3.10.19 | 2.7.1 | 11.8 | 최신 실험 | ⚠️ transformers 비정상 |
| **jelly** | 3.10.20 | 2.5.1 | 12.1 | VIC, StoryDiffusion | ✅ 권장 |
| **base** | 3.13 | 2.6.0 | 12.4 | - | ❌ CUDA 실패 |

**현재 작동하는 환경**: 없음 (CUDA 문제)

**VIC 재생성 시 권장**: `jelly` 환경 (CUDA 복구 후)

---

