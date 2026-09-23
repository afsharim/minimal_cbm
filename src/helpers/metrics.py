import numpy as np
import torch
from torch import Tensor

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def calc_accuracy(preds: Tensor, labels: Tensor):
    if preds.shape[-1]==1:
        accuracy = ((preds[:,0] >= 0.5).float() == labels).float().mean() * 100
    else:
        predicted_labels = torch.argmax(preds, dim=1)
        correct = (predicted_labels == labels).sum().item()
        accuracy = 100 * correct / labels.size(0)
    return accuracy


def get_results_classifier_sklearn(
        x_train: Tensor, 
        y_train: Tensor,
        x_test: Tensor, 
        y_test: Tensor,
        n_layers: int=2,
        max_iter=50,
        **kwargs
    ):
    x_train, y_train = x_train.numpy(), y_train.numpy()
    x_test, y_test = x_test.numpy(), y_test.numpy()
    n_classes = int(y_train.max()+1)
    # Degenerate target (a constant nuisance/concept attribute): it carries no
    # information, so MI is 0 and any classifier is trivially perfect. Guard
    # against divide-by-zero entropy (would otherwise yield NaN).
    if len(np.unique(y_train)) < 2:
        acc = accuracy_score(y_test, np.full_like(y_test, int(y_train[0]))) * 100
        return {'accuracy': acc, 'mi': 0.0}
    if n_layers>0:
        x_dim, y_dim = x_train.shape[-1], int(y_train.max()+1)
        hidden_sizes = tuple(np.linspace(x_dim, y_dim, n_layers+2, dtype=int)[1:-1])
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = MLPClassifier(
            hidden_layer_sizes=hidden_sizes, 
            activation='relu', 
            solver='adam', 
            max_iter=max_iter, 
            batch_size=256,
            random_state=42
        )
    else:
        model = LogisticRegression(
            multi_class='multinomial', 
            solver='lbfgs', 
            max_iter=max_iter
        )
    model.fit(x_train, y_train)
    class_pred = model.predict(x_test)
    y_pred = model.predict_proba(x_test)
    y_pred[y_pred == 0] = 1e-6 
    probs_class = np.bincount(y_train.astype(int), minlength=n_classes) / len(y_train)
    entropy = -(probs_class * np.log(probs_class)).sum()
    cond_entropy = -(y_pred * np.log(y_pred)).sum(axis=1).mean()
    del model
    return {
        'accuracy': accuracy_score(y_test, class_pred) * 100,
        'mi': 1 - cond_entropy / entropy,
    }


def get_results_classifier_torch(
        x_train: Tensor,
        y_train: Tensor,
        x_test: Tensor,
        y_test: Tensor,
        n_layers: int=2,
        max_iter: int=50,
        batch_size: int=256,
        alpha: float=1e-4,
        learning_rate_init: float=1e-3,
        tol: float=1e-4,
        n_iter_no_change: int=10,
        random_state: int=42,
        device='cuda',
        **kwargs
    ):
    """CUDA replica of the sklearn MLPClassifier probe above.

    Mirrors sklearn defaults: Glorot-uniform init (weights and biases),
    Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8), L2 penalty
    0.5*alpha*||W||^2 / batch_size added per batch (weights only),
    per-epoch shuffling, and tol/n_iter_no_change early stopping on the
    training loss. Returns the same {'accuracy', 'mi'} dict.
    """
    if n_layers <= 0:
        return get_results_classifier_sklearn(
            x_train, y_train, x_test, y_test,
            n_layers=n_layers, max_iter=max_iter,
        )
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    x_train = x_train.detach().to(device=device, dtype=torch.float32)
    x_test = x_test.detach().to(device=device, dtype=torch.float32)
    y_train_t = y_train.detach().to(device=device, dtype=torch.long).view(-1)
    y_test_t = y_test.detach().to(device=device, dtype=torch.long).view(-1)

    # Degenerate target: no information -> mi=0, trivial classifier (matches
    # the sklearn branch's guard, avoids divide-by-zero NaN).
    if int(y_train_t.min().item()) == int(y_train_t.max().item()):
        const = int(y_train_t[0].item())
        acc = (y_test_t == const).float().mean().item() * 100
        return {'accuracy': acc, 'mi': 0.0}

    n_classes = int(y_train_t.max().item() + 1)
    x_dim = x_train.shape[-1]
    hidden_sizes = tuple(np.linspace(x_dim, n_classes, n_layers+2, dtype=int)[1:-1])
    # sklearn MLPClassifier uses a single logistic output for binary tasks.
    binary = (n_classes == 2)
    out_dim = 1 if binary else n_classes

    # StandardScaler (biased std, zero-variance features left unscaled)
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False)
    std = torch.where(std == 0, torch.ones_like(std), std)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    gen = torch.Generator(device='cpu').manual_seed(random_state)
    dims = [x_dim] + list(hidden_sizes) + [out_dim]
    weights, biases = [], []
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        bound = float(np.sqrt(6.0 / (fan_in + fan_out)))
        w = (torch.rand(fan_in, fan_out, generator=gen) * 2 - 1) * bound
        b = (torch.rand(fan_out, generator=gen) * 2 - 1) * bound
        weights.append(torch.nn.Parameter(w.to(device)))
        biases.append(torch.nn.Parameter(b.to(device)))
    params = weights + biases

    optimizer = torch.optim.Adam(
        params, lr=learning_rate_init, betas=(0.9, 0.999), eps=1e-8)

    def _forward(x):
        h = x
        for i in range(len(weights) - 1):
            h = torch.relu(h @ weights[i] + biases[i])
        return h @ weights[-1] + biases[-1]

    n_samples = x_train.shape[0]
    best_loss = float('inf')
    no_improvement_count = 0
    for _ in range(max_iter):
        perm = torch.randperm(n_samples, generator=gen).to(device)
        accumulated_loss = 0.0
        for start in range(0, n_samples, batch_size):
            idx = perm[start:start+batch_size]
            xb, yb = x_train[idx], y_train_t[idx]
            logits = _forward(xb)
            if binary:
                ce = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(1), yb.float())
            else:
                ce = torch.nn.functional.cross_entropy(logits, yb)
            l2 = sum((w ** 2).sum() for w in weights)
            loss = ce + 0.5 * alpha * l2 / xb.shape[0]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            accumulated_loss += float(loss.detach()) * xb.shape[0]
        epoch_loss = accumulated_loss / n_samples
        if epoch_loss > best_loss - tol:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        if no_improvement_count > n_iter_no_change:
            break

    with torch.no_grad():
        probs_chunks, pred_chunks = [], []
        for start in range(0, x_test.shape[0], 8192):
            logits = _forward(x_test[start:start+8192])
            if binary:
                p = torch.sigmoid(logits.squeeze(1))
                probs = torch.stack([1 - p, p], dim=1)
                pred = (p >= 0.5).long()
            else:
                probs = torch.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)
            probs_chunks.append(probs)
            pred_chunks.append(pred)
        y_pred = torch.cat(probs_chunks, dim=0)
        class_pred = torch.cat(pred_chunks, dim=0)
        accuracy = (class_pred == y_test_t).float().mean().item() * 100
        y_pred = y_pred.double()
        y_pred[y_pred == 0] = 1e-6
        cond_entropy = -(y_pred * torch.log(y_pred)).sum(dim=1).mean().item()

    y_train_np = y_train.detach().cpu().numpy()
    probs_class = np.bincount(y_train_np.astype(int), minlength=n_classes) / len(y_train_np)
    entropy = -(probs_class * np.log(probs_class)).sum()
    return {
        'accuracy': accuracy,
        'mi': 1 - cond_entropy / entropy,
    }


def get_results_classifier(*args, device='cuda', **kwargs):
    """Dispatch to the CUDA probe unless MCBM_PROBE=sklearn is set."""
    import os
    backend = os.environ.get('MCBM_PROBE', 'torch').lower()
    if backend == 'sklearn' or not torch.cuda.is_available():
        return get_results_classifier_sklearn(*args, **kwargs)
    return get_results_classifier_torch(*args, device=device, **kwargs)


def calc_ece(preds: Tensor, labels: Tensor, n_bins: int = 15):
    confidences, y_hat = preds.max(dim=1)
    accuracies = (y_hat == labels).float()

    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=preds.device)
    ece = torch.tensor(0.0, device=preds.device)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences >= lo) & (confidences <= hi) if i == 0 else (confidences > lo) & (confidences <= hi)
        if in_bin.any():
            prop = in_bin.float().mean()
            ece += (confidences[in_bin].mean() - accuracies[in_bin].mean()).abs() * prop

    return float(ece.item())


def calc_brier(preds, labels) -> float:
    """
    Multiclass Brier score without building one-hot.
    For each sample: sum_k (p_k - y_k)^2 = 1 - 2*p_true + sum_k p_k^2
    """
    p_true = preds.gather(1, labels.view(-1, 1)).squeeze(1)
    sum_p2 = (preds ** 2).sum(dim=1)
    brier = (1.0 - 2.0 * p_true + sum_p2).mean()
    return float(brier.item())



def calc_map(preds, labels) -> float:
    # Flatten
    preds = preds.view(-1)
    labels = labels.view(-1)

    # Sort by predicted score
    sorted_indices = torch.argsort(preds, descending=True)
    sorted_labels = labels[sorted_indices]

    # Cumulative true positives and false positives
    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1 - sorted_labels, dim=0)

    precisions = tp / (tp + fp)
    recalls = tp / labels.sum()

    # If no positives, AP is undefined → return 0
    if labels.sum() == 0:
        return 0.0

    # Interpolated AP = sum over recall steps
    AP = torch.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
    return AP.item()
