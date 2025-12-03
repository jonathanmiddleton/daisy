import torch
import torch.nn.functional as F

def compute_loss_and_tokens(model, batch, vocab_size, device="cuda"):
    device = torch.device(device)

    input_ids = batch["input_ids"].to(device)
    targets = batch["targets"].to(device)

    loss = model(input_ids)

    n_tokens = targets.numel()

    return loss, n_tokens
