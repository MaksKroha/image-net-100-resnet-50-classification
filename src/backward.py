from torch.nn import CrossEntropyLoss


def backward(logits, labels, optimizer, lb_epsi=0.1):
    loss = CrossEntropyLoss(label_smoothing=lb_epsi)(logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()