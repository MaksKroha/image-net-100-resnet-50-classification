from torch.nn import CrossEntropyLoss

from src.utils.logger import exception_logger
from src.utils.analyzer import Analyzer

@exception_logger
def backward(logits, labels, optimizer, lb_epsi=0.1, analyzer=None):
    loss = CrossEntropyLoss(label_smoothing=lb_epsi)(logits, labels)
    loss.backward()
    optimizer.step()
    if analyzer is not None:
        analyzer.add_loss(loss.item())
        analyzer.show_accuracy()

    optimizer.zero_grad()
