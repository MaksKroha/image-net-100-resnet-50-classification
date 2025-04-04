from torch.nn import CrossEntropyLoss

from src.utils.logger import log_exception


def backward(logits, labels, optimizer, lb_epsi=0.1):
    try:
        loss = CrossEntropyLoss(label_smoothing=lb_epsi)(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    except Exception as e:
        log_exception(str(e))