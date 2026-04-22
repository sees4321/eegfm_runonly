import argparse
import time
import json
import torch
import torch.nn as nn
from tabulate import tabulate

# 기존 프로젝트에서 불러오기
from config import EEGModelConfig
from model import EEGEncoder

# FLOPs 측정을 위한 라이브러리 (pip install fvcore)
try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
except ImportError:
    FlopCountAnalysis = None


class ProfileWrapper(nn.Module):
    """
    FLOPs 측정 라이브러리가 forward 함수를 쉽게 추적할 수 있도록
    임베딩 단계와 인코더 단계를 하나로 묶어주는 래퍼 클래스입니다.
    """
    def __init__(self, encoder: EEGEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x, coords, c_idx, t_idx, pad):
        coord_ch = self.encoder.coord_embed(coords)
        tok_ctx, pad_ctx2, rope_ctx, chan_ctx = self.encoder.embed_from_indices(
            x=x, coords=coords, c_idx=c_idx, t_idx=t_idx, pad=pad, coord_ch=coord_ch
        )
        # grid_patches(P_max)는 임의의 값(예: 100)으로 고정하거나 데이터에 맞게 변경
        out = self.encoder(
            tok_ctx, 
            padding_mask=pad_ctx2, 
            rope_pos=rope_ctx, 
            chan_idx=chan_ctx, 
            coords=coords, 
            grid_patches=100 
        )
        return out


def generate_dummy_batch(batch_size: int, num_channels: int, seq_len: int, device: torch.device):
    """
    모델 입력에 맞는 더미 데이터를 생성합니다.
    실제 ShapeBatcher에서 나오는 텐서 모양에 맞게 차원을 수정해야 할 수 있습니다.
    """
    x = torch.randn(batch_size, num_channels, 1000, device=device) # [B, C, L] 가정
    coords = torch.randn(batch_size, num_channels, 3, device=device)
    c_idx = torch.randint(0, num_channels, (batch_size, seq_len), device=device)
    t_idx = torch.randint(0, 100, (batch_size, seq_len), device=device)
    pad = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    
    return x, coords, c_idx, t_idx, pad


def profile_model(config_path: str, batch_size: int, seq_len: int, device: torch.device):
    print(f"\n--- Profiling Config: {config_path} ---")
    
    # 1. 모델 로드
    model_cfg = EEGModelConfig.from_json(config_path)
    model = EEGEncoder(model_cfg).to(device)
    wrapper = ProfileWrapper(model).eval()
    
    # 파라미터 수 계산
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    
    # 더미 데이터 생성 (채널 수는 설정에 맞게 변경)
    num_channels = getattr(model_cfg, 'num_channels', 64) 
    x, coords, c_idx, t_idx, pad = generate_dummy_batch(batch_size, num_channels, seq_len, device)
    inputs = (x, coords, c_idx, t_idx, pad)

    # 2. FLOPs 측정 (fvcore 활용)
    flops_g = 0.0
    if FlopCountAnalysis is not None:
        # Warning 출력을 줄이기 위해 잠시 래퍼 사용
        flops_analyzer = FlopCountAnalysis(wrapper, inputs)
        flops_analyzer.unsupported_ops_warnings(False)
        flops_g = flops_analyzer.total() / 1e9
    else:
        print("fvcore가 설치되지 않아 FLOPs를 계산할 수 없습니다. (pip install fvcore)")

    # 3. Peak Memory & Latency 측정을 위해 학습 모드로 전환 (Gradient 필요)
    wrapper.train()
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler() # Mixed Precision 가정

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = wrapper(*inputs)
            loss = out.sum()
        scaler.scale(loss).backward()
        scaler.step(optimizer)

    # Peak Memory 측정
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        out = wrapper(*inputs)
        loss = out.sum()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    # Latency 측정 (Forward + Backward)
    measure_steps = 50
    torch.cuda.synchronize(device)
    start_time = time.time()
    
    for _ in range(measure_steps):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = wrapper(*inputs)
            loss = out.sum()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        
    torch.cuda.synchronize(device)
    end_time = time.time()
    
    avg_step_ms = ((end_time - start_time) / measure_steps) * 1000
    throughput = (batch_size * seq_len * measure_steps) / (end_time - start_time)

    return {
        "Config": config_path.split("/")[-1],
        "Params (M)": f"{num_params:.2f}",
        "FLOPs (G)": f"{flops_g:.2f}" if flops_g > 0 else "N/A",
        "Peak Mem (GB)": f"{peak_memory_gb:.2f}",
        "Latency (ms/step)": f"{avg_step_ms:.2f}",
        "Throughput (tok/s)": f"{throughput:.0f}"
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True, help="프로파일링할 JSON 설정 파일 목록")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("경고: CUDA가 사용 불가능합니다. 정확한 메모리/속도 측정이 어렵습니다.")

    results = []
    for cfg_path in args.configs:
        try:
            res = profile_model(cfg_path, args.batch_size, args.seq_len, device)
            results.append(res)
        except torch.cuda.OutOfMemoryError:
            results.append({
                "Config": cfg_path.split("/")[-1],
                "Params (M)": "N/A",
                "FLOPs (G)": "N/A",
                "Peak Mem (GB)": "OOM",
                "Latency (ms/step)": "-",
                "Throughput (tok/s)": "-"
            })
            torch.cuda.empty_cache() # 다음 모델을 위해 캐시 비우기
        except Exception as e:
            print(f"Error profiling {cfg_path}: {e}")

    # 마크다운 형태의 예쁜 표로 출력
    print("\n\n=== Profiling Results ===")
    print(tabulate(results, headers="keys", tablefmt="github"))


if __name__ == "__main__":
    main()