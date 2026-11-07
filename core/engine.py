import torch


def TrainEpoch(model, inputs, targets, optimizer, clip_val=1.0):
    """One full-batch training step on the whole training set. Returns (loss, acc)."""
    model.train()

    logits = model(inputs)
    loss, acc = model.computeLoss(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
    optimizer.step()

    return loss.item(), acc


@torch.no_grad()
def Validate(model, inputs, targets):
    """Loss and accuracy on the whole validation set in a single forward pass."""
    model.eval()

    logits = model(inputs)
    loss, acc = model.computeLoss(logits, targets)

    return loss.item(), acc