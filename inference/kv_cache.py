# inference/kv_cache.py
import torch

class KVCache:
    def __init__(self, L, B, H, W, D, device, dtype):
        self.H = H
        self.B = B
        self.D = D
        self.W = W
        self.L = L
        self.t = 0
        self.device = device
        self.dtype = dtype
        self.k = torch.zeros(self.L, self.B, self.H, self.W, self.D, device=self.device, dtype=self.dtype)
        self.v = torch.zeros(self.L, self.B, self.H, self.W, self.D, device=self.device, dtype=self.dtype)
        self.t = 0
        self._staged = [False] * self.L

    def reset_history(self):
        self.k = torch.zeros(self.L, self.B, self.H, self.W, self.D, device=self.device, dtype=self.dtype)
        self.v = torch.zeros(self.L, self.B, self.H, self.W, self.D, device=self.device, dtype=self.dtype)
        self.t = 0
        self._staged = [False] * self.L

#TODO simplify this given that the window is a global property and so this should be a simple tensor op
    def view(self, layer):
        m = self.t + (1 if self._staged[layer] else 0)
        n = min(m, self.W)
        if n == 0:
            zk = self.k[layer, :, :, :0, :]
            zv = self.v[layer, :, :, :0, :]
            return zk.transpose(1, 2), zv.transpose(1, 2)
        i = self.t % self.W
        end = (i + (1 if self._staged[layer] else 0)) % self.W
        start = (end - n) % self.W
        if start < end:
            k = self.k[layer, :, :, start:end, :]
            v = self.v[layer, :, :, start:end, :]
        else:
            k = torch.cat([self.k[layer, :, :, start:, :], self.k[layer, :, :, :end, :]], dim=2)
            v = torch.cat([self.v[layer, :, :, start:, :], self.v[layer, :, :, :end, :]], dim=2)
        return k.transpose(1, 2), v.transpose(1, 2)

    def write(self, layer, k_new, v_new):
        if k_new.size(1) == 1:
            k_new = k_new.transpose(1, 2)
            v_new = v_new.transpose(1, 2)
        i = self.t % self.W
        self.k[layer, :, :, i:i+1, :] = k_new
        self.v[layer, :, :, i:i+1, :] = v_new
        self._staged[layer] = True

    def advance(self):
        self.t += 1
        self._staged = [False] * self.L

    def bulk_write_packed(self, kv, pos):
        k, v = kv[0], kv[1]
        r = min(pos, max(self.W - 1, 0))
        if r == 0:
            self.t = pos
            self._staged = [False] * self.L
            return
        idx = torch.arange(pos - r, pos, device=self.k.device) % self.W
        self.k[:, :, :, idx, :] = k[:, :, :, pos - r:pos, :]
        self.v[:, :, :, idx, :] = v[:, :, :, pos - r:pos, :]
        self.t = pos
        self._staged = [False] * self.L
