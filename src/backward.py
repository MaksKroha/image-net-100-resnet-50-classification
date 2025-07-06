from torch.nn import CrossEntropyLoss

from src.utils.logger import exception_logger

@exception_logger
def backward(logits, labels, optimizer, epoch, lb_epsi=0.1, analyzer=None):
    loss = CrossEntropyLoss(label_smoothing=lb_epsi)(logits, labels)
    loss.backward()
    optimizer.step()
    if analyzer is not None:
        analyzer.add_train_val(loss.item(), epoch)
    print(f"loss - {loss.item()}, epoch - {epoch}")

    optimizer.zero_grad()
