import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.autonotebook import tqdm

from utils.evaluate import Evaluator, get_best_th, save_metrics

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
    for num_batches, batch in enumerate(loader):
        edges.append(batch["binds"].edge_label_index)
        outs = run_update(model, optimizer, batch)
        logits.append(outs[0].detach())
        y_true.append(outs[1].detach())
        loss += float(outs[2].detach().cpu())
    logits = torch.cat(logits)
    y_true = torch.cat(y_true)
    edges = torch.cat(edges, dim=1)
    loss /= (num_batches + 1)
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
def run_test(model, test_loader, th):
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

    y_true = y_true.to(torch.int32)
    e = Evaluator("configs/eval/test_evaluation_params.json")
    edges = torch.stack([sources, targets])
    test_metrics = e.evaluate(logits, y_true, th, edges)

    # save all to results table
    results = pd.DataFrame(sources.cpu().numpy(), columns=["source"])
    results["target"] = targets.cpu().numpy()
    results["score"] = scores.cpu().numpy()
    results["y_pred"] = y_pred.cpu().numpy()
    results["y_true"] = y_true.cpu().numpy()

    results.sort_values(by=["score"], ascending=False, inplace=True)
    results["percentile"] = results.score.rank(pct=True)
    results.set_index(["source", "target"], inplace=True)

    return results, test_metrics


def log_gradients_in_model(model, writer, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            writer.add_histogram(tag + "/grad", value.grad.cpu(), step)


def train_loop(
    model,
    locator,
    train_loader,
    val_loader,
    num_epochs,
    log_gradients=False,
):
    torch.manual_seed(SEED)
    model_params = list(model.parameters())
    optimizer = None
    if model_params:
        optimizer = torch.optim.AdamW(
            model_params,
            lr=locator.config["learning_rate"],
            weight_decay=locator.config["weight_decay"],
        )
    writer = SummaryWriter(
        log_dir=locator.summary_path, comment=locator.config["model_name"]
    )
    ev = Evaluator("configs/eval/test_evaluation_params.json")
    best_metric = float("-inf")
    criteria = locator.config["data_split"] + "_mAP"
    for epoch in tqdm(range(1, num_epochs + 1)):
        logits, y_true, edges, loss = run_train_epoch(model, train_loader, optimizer)
        writer.add_scalar("train/loss", loss, epoch)
        if epoch % locator.config["eval_freq"] > 0:
            continue
        best_th = get_best_th(logits, y_true)
        metrics = ev.evaluate(logits, y_true, best_th, edges)
        for metric, score in metrics.items():
            writer.add_scalar("train/" + metric, score, epoch)
        writer.flush()

        logits, y_true, edges = run_inference_epoch(model, val_loader)
        metrics = ev.evaluate(logits, y_true, best_th, edges)
        for metric, score in metrics.items():
            writer.add_scalar("valid/" + metric, score, epoch)
        writer.flush()

        if metrics[criteria] > best_metric:
            best_metric = metrics[criteria]
            state = dict(
                model_state_dict=model.state_dict(),
                best_th=best_th,
            )
            torch.save(state, locator.model_path)
            save_metrics(metrics, locator.valid_metrics_path)

        if log_gradients:
            log_gradients_in_model(model, writer, epoch)

    best_params = torch.load(locator.model_path, weights_only=True)
    best_th = best_params["best_th"]
    model.load_state_dict(best_params["model_state_dict"])
    print(f"Best {criteria}: " + str(best_metric))
    return best_th
