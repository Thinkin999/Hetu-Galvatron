import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
import os

def main():
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get('LOCAL_RANK', 0))  # 添加默认值以避免 KeyError
    torch.cuda.set_device(local_rank)
    
    # # 使用简化的 profiler 配置
    # torch_profiler = profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=False,  # 禁用堆栈跟踪
    #     with_modules=False  # 禁用模块信息
    # )
    # torch_profiler.__enter__()
    
    for step in range(5):
        A = torch.randn((512, 512), device=torch.cuda.current_device())
        B = torch.randn((512, 512), device=torch.cuda.current_device())
        COMM = torch.ones_like(A, device=torch.cuda.current_device())
        
        # 使用同步操作
        dist.all_reduce(COMM)
        
        C = torch.mm(A, B)
        
        #torch_profiler.step()
        print(f"Step {step} completed")
    
    # torch_profiler.__exit__(None, None, None)
    
    # # 使用绝对路径并添加 rank 信息，避免多进程覆盖同一文件
    # output_dir = os.path.join(os.getcwd(), "traces")  # 使用当前工作目录
    # os.makedirs(output_dir, exist_ok=True)
    # output_file = os.path.join(output_dir, f"trace_rank{local_rank}.json")
    # torch_profiler.export_chrome_trace(output_file)
    # print(f"Rank {local_rank}: Trace saved to {output_file}")

if __name__ == "__main__":
    main()
