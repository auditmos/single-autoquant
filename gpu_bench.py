import torch, time
torch.backends.cudnn.benchmark = True
dev = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")
x = torch.randn(512, 168, 23, device=dev)
m = torch.nn.LSTM(23, 384, 3, batch_first=True, dropout=0.3).to(dev)
m.train()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(27):
    out, _ = m(x)
    out.sum().backward()
torch.cuda.synchronize()
print(f"LSTM 27 batches (=1 epoch): {time.time()-t0:.2f}s")
a = torch.randn(512, 384, device=dev)
b = torch.randn(384, 384, device=dev)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(1000):
    c = a @ b
torch.cuda.synchronize()
print(f"matmul 1000x: {time.time()-t0:.2f}s")
