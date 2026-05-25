import torch

def TrainEpoch(model, loader, optimizer, device, clip_val=1.0):
    model.train()
    total_acc = 0.0
    
    for inputs, outputs in loader:
        inputs, outputs = inputs.to(device), outputs.to(device)
        
        logits = model(inputs)
        loss, acc = model.computeLoss(logits, outputs)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
        optimizer.step()
        
        total_acc += acc
        
    return total_acc / len(loader)

@torch.no_grad()
def Validate(model, loader, device):
    model.eval()
    total_acc = 0.0
    
    for inputs, outputs in loader:
        inputs, outputs = inputs.to(device), outputs.to(device)
        
        logits = model(inputs)
        _, acc = model.computeLoss(logits, outputs)
        
        total_acc += acc
        
    return total_acc / len(loader)