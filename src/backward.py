from torch.nn import CrossEntropyLoss

from src.utils.logger import log_exception
from src.utils.analizer import Analizer


def backward(logits, labels, optimizer, lb_epsi=0.1, analizer=None):
    try:
        loss = CrossEntropyLoss(label_smoothing=lb_epsi)(logits, labels)
        loss.backward()
        optimizer.step()
        if analizer is not None:
            analizer.add_loss(loss.item())
            analizer.show_accuracy()
            
        optimizer.zero_grad()
    except Exception as e:
        print(str(e))
        log_exception(str(e))