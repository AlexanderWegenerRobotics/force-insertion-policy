import torch
from ddpm import NoiseEstimator

model = NoiseEstimator(hidden_dim=512)
model.load_state_dict(torch.load("archive/tacDiffusionBase/df3/best.pt", map_location="cpu"))
model.eval()

# Dummy inputs matching the forward signature
obs_prev     = torch.randn(1, 18)
obs_curr     = torch.randn(1, 18)
action_noisy = torch.randn(1, 6)
tau          = torch.tensor([25])

torch.onnx.export(
    model,
    (obs_prev, obs_curr, action_noisy, tau),
    "noise_estimator.onnx",
    input_names=["obs_prev", "obs_curr", "action_noisy", "tau"],
    output_names=["eps_hat"],
    dynamic_axes={"obs_prev": {0: "batch"}, "obs_curr": {0: "batch"},
                  "action_noisy": {0: "batch"}, "tau": {0: "batch"}},
    opset_version=17,
)