from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.autonotebook import tqdm

SEED = 2024313


def run_update(model, optimizer, data):
    if optimizer:  # with Cosine model, optimizer is None
        optimizer.zero_grad()
    logits = model(data)
    y_true = data["binds"].edge_label
    paired = data.bpr_indices
    positives, negatives = logits[paired].T
    num_nodes = data.bpr_weights.sum()
    errors = -F.logsigmoid(positives - negatives) * data.bpr_weights
    loss = errors.sum() / num_nodes
    if optimizer:  # with Cosine model, optimizer is None
        loss.backward()
        optimizer.step()
    return logits, y_true, loss


def run_train_epoch(model, loader, optimizer):
    model.train()
    edges, logits, y_true = [], [], []
    loss = 0
    for num_batches, batch in enumerate(loader, 1):
        edges.append(batch["binds"].edge_label_index)
        outs = run_update(model, optimizer, batch)
        logits.append(outs[0].detach())
        y_true.append(outs[1].detach())
        loss += float(outs[2].detach().cpu())
    logits = torch.cat(logits)
    y_true = torch.cat(y_true)
    edges = torch.cat(edges, dim=1)
    loss /= num_batches
    return logits, y_true, edges, loss


@torch.inference_mode
def run_inference_epoch(model, loader):
    logits = []
    y_true = []
    edges = []
    model.eval()
    for batch in loader:
        logits.append(model(batch))
        y_true.append(batch["binds"].edge_label)
        edges.append(batch["binds"].edge_label_index)
    logits = torch.cat(logits, dim=0)
    y_true = torch.cat(y_true, dim=0)
    edges = torch.cat(edges, dim=1)
    return logits, y_true, edges


@torch.inference_mode
def compute_loss(model, loader):
    loss = 0
    model.eval()
    for num_batches, batch in enumerate(loader, 1):
        logits = model(batch)
        paired = batch.bpr_indices
        positives, negatives = logits[paired].T
        num_nodes = batch.bpr_weights.sum()
        errors = -F.logsigmoid(positives - negatives) * batch.bpr_weights
        loss += errors.sum() / num_nodes
    loss /= num_batches
    return loss


@torch.inference_mode
def run_test(model, test_loader, th=0):
    logits = []
    y_true = []
    src_ids = []
    tgt_ids = []
    model.eval()
    for batch in test_loader:
        logits.append(model(batch))
        y_true.append(batch["binds"].edge_label)

        # sampled batch source and target ids of each edge in test set
        test_srcs = batch["binds"].edge_label_index[0]
        test_tgts = batch["binds"].edge_label_index[1]

        # global source and target ids of each edge in test set
        src_ids.append(batch["source"].node_id[test_srcs])
        tgt_ids.append(batch["target"].node_id[test_tgts])

    # save logits, scores, bool predictions, gt, and indices of srcs and tgts
    logits = torch.cat(logits, dim=0)
    y_true = torch.cat(y_true, dim=0)
    y_pred = logits > th
    scores = torch.sigmoid(logits)
    sources = torch.cat(src_ids, dim=0)
    targets = torch.cat(tgt_ids, dim=0)

    # save all to results table
    results = pd.DataFrame(sources.cpu().numpy(), columns=["source"])
    results["target"] = targets.cpu().numpy()
    results["score"] = scores.cpu().numpy()
    results["logits"] = logits.cpu().numpy()
    results["y_pred"] = y_pred.cpu().numpy()
    results["y_true"] = y_true.cpu().numpy().astype(bool)

    return results


def log_gradients_in_model(model, writer, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            writer.add_histogram(tag + "/grad", value.grad.cpu(), step)


def train_loop(
    model,
    model_path,
    config,
    train_loader,
    valid_loader,
    log_gradients=False,
):
    torch.manual_seed(SEED)
    model_params = list(model.parameters())
    optimizer = None
    if model_params:
        optimizer = torch.optim.AdamW(
            model_params,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    summary_path = Path(model_path).parent
    writer = SummaryWriter(log_dir=summary_path, comment=config["model"])
    best_loss = float("inf")
    for epoch in tqdm(range(1, config["num_epochs"] + 1)):
        logits, y_true, edges, loss = run_train_epoch(model, train_loader, optimizer)
        writer.add_scalar("train/loss", loss, epoch)
        if epoch % config["eval_freq"] == 0:
            loss = compute_loss(model, valid_loader)
            writer.add_scalar("valid/loss", loss, epoch)
            writer.flush()
            if loss < best_loss:
                best_loss = loss
                state = dict(
                    model_state_dict=model.state_dict(),
                )
                torch.save(state, model_path)
            if log_gradients:
                log_gradients_in_model(model, writer, epoch)

    best_params = torch.load(model_path, weights_only=True)
    model.load_state_dict(best_params["model_state_dict"])
