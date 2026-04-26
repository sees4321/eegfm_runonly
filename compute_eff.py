import argparse
import time
import torch
import torch.nn as nn
from tabulate import tabulate
from pathlib import Path

# 기존 프로젝트에서 불러오기
from .config import EEGModelConfig
from .model import EEGEncoder

# FLOPs 측정을 위한 라이브러리 (pip install fvcore)
try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None

# python profile_dir.py --config_dir ./my_configs --batch_size 16 --seq_len 1024

class ProfileWrapper(nn.Module):
    def __init__(self, encoder: EEGEncoder, grid_patches: int):
        super().__init__()
        self.encoder = encoder
        self.grid_patches = grid_patches # 동적으로 받도록 수정

    def forward(self, x, coords, c_idx, t_idx, pad):
        coord_ch = self.encoder.coord_embed(coords)
        tok_ctx, pad_ctx2, rope_ctx, chan_ctx = self.encoder.embed_from_indices(
            x=x, coords=coords, c_idx=c_idx, t_idx=t_idx, pad=pad, coord_ch=coord_ch
        )
        out = self.encoder(
            tok_ctx, 
            padding_mask=pad_ctx2, 
            rope_pos=rope_ctx, 
            chan_idx=chan_ctx, 
            coords=coords, 
            grid_patches=self.grid_patches # 수정됨
        )
        return out


def generate_dummy_batch(batch_size: int, num_channels: int, seq_len: int, device: torch.device):
    x = torch.randn(batch_size, num_channels, seq_len//num_channels*200, device=device)
    coords = torch.randn(batch_size, num_channels, 3, device=device)
    c_idx = torch.randint(0, num_channels, (batch_size, seq_len), device=device)
    t_idx = torch.randint(0, seq_len//num_channels, (batch_size, seq_len), device=device)
    pad = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    
    return x, coords, c_idx, t_idx, pad


def profile_model(config_path: Path, batch_size: int, seq_len: int, device: torch.device):
    print(f"\n--- Profiling Config: {config_path.name} ---")
    
    model_cfg = EEGModelConfig.from_json(str(config_path))
    model = EEGEncoder(model_cfg).to(device)
    num_channels = getattr(model_cfg, 'num_channels', 64) 

    wrapper = ProfileWrapper(model, seq_len//num_channels).eval()
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    
    x, coords, c_idx, t_idx, pad = generate_dummy_batch(batch_size, num_channels, seq_len, device)
    inputs = (x, coords, c_idx, t_idx, pad)

    flops_g = 0.0
    if FlopCountAnalysis is not None:
        flops_analyzer = FlopCountAnalysis(wrapper, inputs)
        flops_analyzer.unsupported_ops_warnings(False)
        flops_g = flops_analyzer.total() / 1e9

    wrapper.train()
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-4)
    
    # 1. 최신 PyTorch 문법으로 GradScaler 초기화 (경고 해결)
    scaler = torch.amp.GradScaler('cuda')

    # 2. Warmup
    for _ in range(5):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'): # 최신 문법 적용
            out = wrapper(*inputs)
            loss = out.sum()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() # ✅ 에러의 원인 해결: 반드시 update()를 호출해야 함!

    # 3. Peak Memory 측정
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad()
    with torch.amp.autocast('cuda'):
        out = wrapper(*inputs)
        loss = out.sum()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update() # ✅ 여기도 추가
    
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    # 4. Latency 측정
    measure_steps = 50
    torch.cuda.synchronize(device)
    start_time = time.time()
    
    for _ in range(measure_steps):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            out = wrapper(*inputs)
            loss = out.sum()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() # ✅ 여기도 추가
        
    torch.cuda.synchronize(device)
    end_time = time.time()
    
    avg_step_ms = ((end_time - start_time) / measure_steps) * 1000
    throughput = (batch_size * seq_len * measure_steps) / (end_time - start_time)

    return {
        "Config": config_path.name,
        "Params (M)": f"{num_params:.2f}",
        "FLOPs (G)": f"{flops_g:.2f}" if flops_g > 0 else "N/A",
        "Peak Mem (GB)": f"{peak_memory_gb:.2f}",
        "Latency (ms/step)": f"{avg_step_ms:.2f}",
        "Throughput (tok/s)": f"{throughput:.0f}"
    }


def main():
    parser = argparse.ArgumentParser()
    # configs 리스트 대신 폴더 경로 하나만 받도록 수정
    parser.add_argument("--config_dir", type=str, required=True, help="JSON 설정 파일들이 들어있는 폴더 경로")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=3840)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("경고: CUDA가 사용 불가능합니다. 정확한 메모리/속도 측정이 어렵습니다.")

    config_dir = Path(args.config_dir)
    if not config_dir.exists() or not config_dir.is_dir():
        print(f"오류: '{args.config_dir}' 폴더를 찾을 수 없습니다.")
        return

    # 지정한 폴더 및 하위 폴더에 있는 모든 .json 파일을 리스트로 만듦
    json_files = sorted(list(config_dir.rglob("*.json")))

    if not json_files:
        print(f"'{args.config_dir}' 폴더 내에 .json 파일이 없습니다.")
        return

    print(f"총 {len(json_files)}개의 설정 파일을 찾았습니다. 프로파일링을 시작합니다...\n")

    results = []
    for cfg_path in json_files:
        try:
            res = profile_model(cfg_path, args.batch_size, args.seq_len, device)
            results.append(res)
        except torch.cuda.OutOfMemoryError:
            results.append({
                "Config": cfg_path.name,
                "Params (M)": "N/A",
                "FLOPs (G)": "N/A",
                "Peak Mem (GB)": "OOM",
                "Latency (ms/step)": "-",
                "Throughput (tok/s)": "-"
            })
            torch.cuda.empty_cache() # 다음 모델 측정을 위해 캐시 비우기
        except Exception as e:
            print(f"Error profiling {cfg_path.name}: {e}")

    # 최종 결과 출력
    print("\n\n=== Profiling Results ===")
    print(tabulate(results, headers="keys", tablefmt="github"))


if __name__ == "__main__":
    main()