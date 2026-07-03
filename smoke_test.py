# Local smoke test for the stability fixes: runs the full training math
# (G forward -> D update -> G update) on tiny tensors, with an untrained VGG
# so no 528MB weight download is needed. Not part of the repo.
import sys
import types
import torch
import torchvision.models as tvm

# stub the pretrained VGG: same architecture, random weights
_real_vgg16 = tvm.vgg16
tvm.vgg16 = lambda *a, **k: _real_vgg16(weights=None)

# cv2 is only used in dataset.py at runtime; stub so imports work without opencv
sys.modules.setdefault('cv2', types.ModuleType('cv2'))
mpl = types.ModuleType('matplotlib'); mpl.pyplot = types.ModuleType('pyplot')
sys.modules.setdefault('matplotlib', mpl)
sys.modules.setdefault('matplotlib.pyplot', mpl.pyplot)
tqdm_mod = types.ModuleType('tqdm'); tqdm_auto = types.ModuleType('tqdm.auto')
tqdm_auto.tqdm = lambda x, **k: x; tqdm_mod.auto = tqdm_auto
sys.modules.setdefault('tqdm', tqdm_mod)
sys.modules.setdefault('tqdm.auto', tqdm_auto)

from models import Generator, Discriminator
from losses import total_gen_loss, total_disc_loss
from utils import gen_to_comp

torch.manual_seed(0)
B, HW = 2, 128  # 128 = 2^7, matches the 7 stride-2 encoder blocks

gen = Generator(9, hidden_in=8)
disc = Discriminator(hidden_channels=8)
gen_loss_fn = total_gen_loss(disc)
disc_loss_fn = total_disc_loss(disc)

gt = torch.rand(B, 3, HW, HW) * 2 - 1
mask = (torch.rand(B, 1, HW, HW) > 0.7).float()
sketch = torch.rand(B, 1, HW, HW)
gen_input = torch.cat([gt * (1 - mask), mask, sketch, torch.rand(B, 3, HW, HW) * 2 - 1, torch.rand(B, 1, HW, HW)], dim=1)
assert gen_input.shape[1] == 9

Igen = gen(gen_input)[0]
assert Igen.shape == (B, 3, HW, HW), Igen.shape
assert Igen.min() >= -1 and Igen.max() <= 1, "tanh range violated"
Icomp = gen_to_comp(mask, Igen, gt)
assert torch.equal(Icomp * (1 - mask), gt * (1 - mask)), "comp must keep gt outside the mask"

# D output must be unbounded (no sigmoid)
with torch.no_grad():
    d_out = disc(torch.cat((gt, mask, sketch), dim=1), five_channels=True)
print(f"D(real) range: [{d_out.min():.3f}, {d_out.max():.3f}] (unbounded critic OK)")

# discriminator update
disc_loss = disc_loss_fn(mask, sketch, Icomp, gt)
disc_loss.backward()  # no retain_graph
assert torch.isfinite(disc_loss), disc_loss
d_grads = [p.grad for p in disc.parameters() if p.grad is not None]
assert len(d_grads) > 0 and all(torch.isfinite(g).all() for g in d_grads)
g_grad_after_d = [p.grad for p in gen.parameters() if p.grad is not None]
assert len(g_grad_after_d) == 0, "disc update must not touch the generator graph"
print(f"disc_loss = {disc_loss.item():.4f}, {len(d_grads)} disc params got finite grads")

# generator update (reuses Igen's graph)
gen_loss = gen_loss_fn(mask, sketch, Igen, Icomp, gt)
gen_loss.backward()
assert torch.isfinite(gen_loss), gen_loss
g_grads = [p.grad for p in gen.parameters() if p.grad is not None]
assert len(g_grads) > 0 and all(torch.isfinite(g).all() for g in g_grads)
print(f"gen_loss  = {gen_loss.item():.4f}, {len(g_grads)} gen params got finite grads")

# individual gen-loss components should now be on comparable scales
pp = gen_loss_fn.per_pixel_loss(mask, Igen, gt)
tv = gen_loss_fn.tv_loss(Icomp, mask)
percep, style = gen_loss_fn.percept_style_losses(gt, Igen)
adv = gen_loss_fn.gen_adv_loss(mask, sketch, Icomp, gt)
print(f"components: per_pixel={pp.item():.4f} percep={percep.item():.4f} "
      f"style={style.item():.6f} tv={tv.item():.4f} adv={adv.item():.6f}")

print("SMOKE TEST PASSED")
