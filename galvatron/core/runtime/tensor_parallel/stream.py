import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.distributed as dist

def run():
    rank = dist.get_rank()
    size = dist.get_world_size()
    tensor_send = torch.zeros(1, device=f'cuda:{rank}')
    tensor_recv = torch.zeros(1, device=f'cuda:{rank}')

    send_stream = torch.cuda.Stream(device=f'cuda:{rank}')
    recv_stream = torch.cuda.Stream(device=f'cuda:{rank}')
    
    if rank == 0:
        tensor_send += 2 
        with torch.cuda.stream(send_stream):
            req1 = dist.isend(tensor=tensor_send, dst=1)
        with torch.cuda.stream(recv_stream):
            req2 = dist.irecv(tensor=tensor_recv, src=1)
        req1.wait() 
        print("req1 finished")
        req2.wait() # will hang here
        print('Rank 0 received:', tensor_recv.item())
    elif rank == 1:
        tensor_send += 3
        with torch.cuda.stream(recv_stream):
            req2 = dist.irecv(tensor=tensor_recv, src=0)
        with torch.cuda.stream(send_stream):
            req1 = dist.isend(tensor=tensor_send, dst=0)
        req1.wait()  
        print("req1 finished")
        req2.wait() # will hang here
        print('Rank 1 received:', tensor_recv.item())

def profile_stream_usage():
    x = torch.randn(1000, 1000, device='cuda')
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True
    ) as prof:
        # with record_function("## Computation"):
        #     y = x @ x.t()
        #     print(torch.cuda.current_stream())
        
        # with record_function("## Memory Copy"):
        #     #z = y.cpu()
        #     z = y.to('cpu', non_blocking=True)
        # with record_function("##create_stream"):
        #     stream1 = torch.cuda.Stream()
        #     stream2 = torch.cuda.Stream()
            
        #     a = torch.randn(1000, 1000, device='cuda')
        #     b = torch.randn(1000, 1000, device='cuda')
        #     with torch.cuda.stream(stream1):
        #         c = a @ b.t()
        #         print(torch.cuda.current_stream())
        #     with torch.cuda.stream(stream2):
        #         d = c.to('cpu', non_blocking=True)
        #         print(torch.cuda.current_stream())
        with record_function("##start run"):
            run()
            
    rank = dist.get_rank()
    # 导出为Chrome tracing格式
    if rank % 2 == 0:
        prof.export_chrome_trace("stream_trace_05.json")
    else:
        prof.export_chrome_trace("stream_trace_06.json")

#profile_stream_usage()
if __name__ == "__main__":
    dist.init_process_group("nccl", init_method='env://', world_size=2)  # init nccl
    profile_stream_usage()