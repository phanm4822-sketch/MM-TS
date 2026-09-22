"""Training, validation and testing for MM-TS."""

import copy
import json
import os
import time

import torch
from data_provider.data_factory import data_provider, is_electricity_manifest, load_cache_metadata
from exp.exp_basic import Exp_Basic
from models.MMTS import Model
from utils.tools import ensure_dir
from utils.config import MODEL_RECIPE
from utils.ts_attention import validate_checkpoint_attention, validate_checkpoint_forecasting
from utils.prompts import UNIFIED_PROMPT_VERSION


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        self.dataset_name = str(getattr(args, "dataset_name", "") or "")
        self.data_mean = self.data_std = None
        super().__init__(args)

    def _build_model(self):
        ensure_dir(self.args.output_dir)
        return Model(self.args, device=self.device)

    def _build_metrics_paths(self):
        dataset_name = str(getattr(self, "dataset_name", "") or "").strip() or "dataset"
        dataset_dir = os.path.join(self.args.output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        run_tag = time.strftime("%Y%m%d_%H%M%S")
        unique_path = os.path.join(dataset_dir, f"metrics_{run_tag}_{os.getpid()}.json")
        latest_path = os.path.join(dataset_dir, "metrics.latest.json")
        return unique_path, latest_path

    def _write_metrics_latest(
        self,
        latest_path: str,
        metrics: dict,
        best_epoch: int,
        best_monitor: float,
        status: str,
        current_epoch: int,
        final_test_mse: float | None = None,
    ):
        run_info = {
            "best_epoch": int(best_epoch),
            "best_val_mse": float(best_monitor),
            "monitor_target": "val",
            "status": str(status),
            "current_epoch": int(current_epoch),
            "args": self._collect_run_config(),
        }
        if final_test_mse is not None:
            run_info["final_test_mse"] = float(final_test_mse)
        payload = dict(metrics)
        payload["run_info"] = run_info
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _build_checkpoint_paths(self):
        dataset_name = str(getattr(self, "dataset_name", "") or "").strip() or "dataset"
        ckpt_dir = os.path.join(self.args.output_dir, dataset_name, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        run_tag = time.strftime("%Y%m%d_%H%M%S")
        unique_path = os.path.join(ckpt_dir, f"best_{run_tag}_{os.getpid()}.pt")
        latest_path = os.path.join(ckpt_dir, "best.latest.pt")
        return unique_path, latest_path

    def _capture_checkpoint_state(self, best_epoch: int, best_monitor: float):
        state = {
            "best_epoch": int(best_epoch),
            "best_val_mse": float(best_monitor),
            "args": self._collect_run_config(),
            "ts_mlp": self._state_dict_to_cpu(self.model.ts_mlp.state_dict()),
            "pred_head": self._state_dict_to_cpu(self.model.pred_head.state_dict()),
        }
        if self.model is not None:
            state["model"] = self._state_dict_to_cpu(self.model.vlm.state_dict())
        if getattr(self.model, "ts_time_embed", None) is not None:
            state["ts_time_embed"] = self._state_dict_to_cpu(self.model.ts_time_embed.state_dict())
        if getattr(self.model, "ts_var_embed", None) is not None:
            state["ts_var_embed"] = self._state_dict_to_cpu(self.model.ts_var_embed.state_dict())
        if getattr(self.model, "ts_normalizer", None) is not None:
            state["ts_normalizer"] = self._state_dict_to_cpu(self.model.ts_normalizer.state_dict())
        return state

    @staticmethod
    def _state_dict_to_cpu(state_dict: dict):
        cpu_state = {}
        for key, value in state_dict.items():
            if torch.is_tensor(value):
                cpu_state[key] = value.detach().cpu().clone()
            else:
                cpu_state[key] = copy.deepcopy(value)
        return cpu_state

    def _restore_checkpoint_state(self, state: dict):
        validate_checkpoint_attention(self.args, state.get("args", {}))
        validate_checkpoint_forecasting(self.args, state.get("args", {}))
        self.model.ts_mlp.load_state_dict(state["ts_mlp"])
        self.model.pred_head.load_state_dict(state["pred_head"])
        if "model" in state:
            self.model.vlm.load_state_dict(state["model"], strict=True)
        if "ts_time_embed" in state and getattr(self.model, "ts_time_embed", None) is not None:
            self.model.ts_time_embed.load_state_dict(state["ts_time_embed"])
        if "ts_var_embed" in state and getattr(self.model, "ts_var_embed", None) is not None:
            self.model.ts_var_embed.load_state_dict(state["ts_var_embed"])
        if "ts_normalizer" in state and getattr(self.model, "ts_normalizer", None) is not None:
            self.model.ts_normalizer.load_state_dict(state["ts_normalizer"])

    def save_checkpoint(
        self, path: str, best_epoch: int, best_monitor: float, state: dict | None = None
    ):
        if state is None:
            state = self._capture_checkpoint_state(best_epoch=best_epoch, best_monitor=best_monitor)
        torch.save(state, path)
        print(f"[Checkpoint] saved best checkpoint -> {path}")

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"checkpoint not found: {path}")
        _train_data, _train_loader = self._get_data(flag="train")
        state = torch.load(path, map_location="cpu")
        self._restore_checkpoint_state(state)
        print(
            f"[Checkpoint] loaded -> {path} "
            f"(best_epoch={state.get('best_epoch', 'na')} "
            f"best_val_mse={state.get('best_val_mse', 'na')})"
        )

    def _collect_run_config(self):
        cfg = {}
        for k, v in vars(self.args).items():
            if k.startswith("_"):
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                cfg[k] = v
            elif isinstance(v, (list, tuple)):
                cfg[k] = list(v)
            elif isinstance(v, dict):
                cfg[k] = v
            else:
                cfg[k] = str(v)
        cfg.update(MODEL_RECIPE)
        if cfg.get("prompt_style") == "unified":
            cfg["prompt_template_version"] = UNIFIED_PROMPT_VERSION
        return cfg

    def save_eval_metrics(self, test_metrics: dict, ckpt_path: str = ""):
        save_path, latest_path = self._build_metrics_paths()
        payload = {
            "test": [test_metrics],
            "run_info": {
                "mode": "eval_only",
                "ckpt_path": str(ckpt_path or ""),
                "args": self._collect_run_config(),
            },
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[Output] metrics saved to: {save_path}")
        print(f"[Output] latest metrics: {latest_path}")
        return save_path, latest_path

    def _get_data(self, flag):
        if flag == "train":
            data_path = getattr(self.args, "data_path", "unknown")
            if is_electricity_manifest(str(data_path)):
                source = "electricity_shards"
            else:
                source = "npz" if str(data_path).endswith(".npz") else "csv"
            print(f"[Data] source={source} path={data_path}")
        data_set, data_loader = data_provider(self.args, flag=flag)
        if flag == "train":
            data_path = getattr(self.args, "data_path", None)
            if data_path:
                try:
                    cache_meta = load_cache_metadata(data_path)
                    meta = cache_meta.get("meta", {})
                    source_dataset_name = str(meta.get("source_dataset_name", "") or "").strip()
                    if source_dataset_name:
                        self.dataset_name = source_dataset_name
                    if cache_meta.get("mean") is not None and cache_meta.get("std") is not None:
                        self.data_mean = torch.as_tensor(cache_meta["mean"], dtype=torch.float32)
                        self.data_std = torch.as_tensor(cache_meta["std"], dtype=torch.float32)
                except Exception:
                    self.data_mean = None
                    self.data_std = None
            self.model.dataset_name = self.dataset_name
            self.model.configure_channels()

        if flag == "train" and hasattr(data_set, "split_sizes"):
            split_sizes = data_set.split_sizes
            print(
                f"[Dataset] split=train windows={len(data_set)} "
                f"(train={split_sizes[0]}, val={split_sizes[1]}, test={split_sizes[2]})"
            )
        return data_set, data_loader

    def _select_optimizer(self):
        train_params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(train_params, lr=self.args.lr, weight_decay=self.args.weight_decay)

    def _select_scheduler(self, optimizer):
        schedule = str(getattr(self.args, "lr_schedule", "none")).strip().lower()
        if schedule == "none":
            return None
        if schedule == "cosine":
            total_epochs = max(1, int(getattr(self.args, "epochs", 1)))
            min_lr_ratio = float(getattr(self.args, "min_lr_ratio", 0.1))
            min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
                eta_min=float(getattr(self.args, "lr", 0.0)) * min_lr_ratio,
            )
        raise ValueError(f"unknown lr_schedule: {schedule}")

    @staticmethod
    def _has_trainable_params(module) -> bool:
        return any(p.requires_grad for p in module.parameters())

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 8:
                return batch
            if len(batch) == 4:
                x, y, img_grids, vid_grids = batch
                return x, y, img_grids, vid_grids, None, None, None, None
        x, y = batch
        return x, y, None, None, None, None, None, None

    @torch.no_grad()
    def vali(self, vali_loader):
        self.model.ts_mlp.eval()
        self.model.pred_head.eval()
        self.model.eval()
        self.model.backbone.eval()

        total_sq = 0.0
        total_abs = 0.0
        total_count = 0

        max_batches = self.args.max_eval_batches
        for batch_idx, batch in enumerate(vali_loader):
            x, y, img_grids, vid_grids, img_tokens, img_token_mask, vid_tokens, vid_token_mask = (
                self._unpack_batch(batch)
            )

            preds = self.model(
                x=x,
                img_grids=img_grids,
                vid_grids=vid_grids,
                img_tokens=img_tokens,
                img_token_mask=img_token_mask,
                vid_tokens=vid_tokens,
                vid_token_mask=vid_token_mask,
            )
            y_dev = y.to(preds.device)

            preds_eval, y_eval = self._maybe_denorm(preds, y_dev)
            diff = preds_eval - y_eval
            total_sq += float(torch.sum(diff**2).item())
            total_abs += float(torch.sum(torch.abs(diff)).item())
            total_count += int(y_eval.numel())

            if max_batches is not None and max_batches > 0 and (batch_idx + 1) >= max_batches:
                break

        mse_epoch = total_sq / max(1, total_count)
        mae_epoch = total_abs / max(1, total_count)
        loss_epoch = total_sq / max(1, total_count)
        if bool(getattr(self.args, "print_per_var_metrics", False)):
            per_var = self._per_var_metrics(vali_loader)
            print(f"[Val] per-var mse={per_var['mse']} mae={per_var['mae']}")
        return {"mse": mse_epoch, "mae": mae_epoch, "loss": loss_epoch}

    def train(self):
        start_time = time.perf_counter()
        _train_data, train_loader = self._get_data(flag="train")
        _vali_data, vali_loader = self._get_data(flag="val")

        optimizer = self._select_optimizer()
        scheduler = self._select_scheduler(optimizer)

        metrics = {"train": [], "val": [], "test_epoch": [], "test": []}
        save_path, latest_path = self._build_metrics_paths()
        best_monitor = float("inf")
        best_epoch = -1
        best_state = None
        bad_epochs = 0
        patience = int(getattr(self.args, "patience", 0))
        eval_test_during_train = bool(getattr(self.args, "eval_test_during_train", False))
        test_loader = None
        if eval_test_during_train:
            _test_data, test_loader = self._get_data(flag="test")
        for epoch in range(self.args.epochs):
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            self.model.ts_mlp.train()
            self.model.pred_head.train()
            if self._has_trainable_params(self.model):
                self.model.train()
            else:
                self.model.eval()
            if self._has_trainable_params(self.model.backbone):
                self.model.backbone.train()
            else:
                self.model.backbone.eval()

            total_sq = 0.0
            total_abs = 0.0
            total_count = 0

            max_batches = self.args.max_train_batches
            for batch_idx, batch in enumerate(train_loader):
                (
                    x,
                    y,
                    img_grids,
                    vid_grids,
                    img_tokens,
                    img_token_mask,
                    vid_tokens,
                    vid_token_mask,
                ) = self._unpack_batch(batch)
                optimizer.zero_grad(set_to_none=True)

                preds = self.model(
                    x=x,
                    img_grids=img_grids,
                    vid_grids=vid_grids,
                    img_tokens=img_tokens,
                    img_token_mask=img_token_mask,
                    vid_tokens=vid_tokens,
                    vid_token_mask=vid_token_mask,
                )
                y_dev = y.to(preds.device)

                diff = preds - y_dev
                mse = torch.mean(diff**2)
                mse.backward()
                optimizer.step()

                total_sq += float(torch.sum(diff**2).item())
                total_abs += float(torch.sum(torch.abs(diff)).item())
                total_count += int(y_dev.numel())

                if (batch_idx + 1) % max(1, self.args.log_interval) == 0:
                    print(f"[train] epoch={epoch + 1} step={batch_idx + 1} mse={mse.item():.6f}")

                if max_batches is not None and max_batches > 0 and (batch_idx + 1) >= max_batches:
                    break

            train_metrics = {
                "mse": total_sq / max(1, total_count),
                "mae": total_abs / max(1, total_count),
                "loss": total_sq / max(1, total_count),
            }
            val_metrics = self.vali(vali_loader)
            metrics["train"].append(train_metrics)
            metrics["val"].append(val_metrics)
            if eval_test_during_train and test_loader is not None:
                test_metrics = self.vali(test_loader)
                metrics["test_epoch"].append(test_metrics)
                print(
                    f"[Epoch {epoch + 1}] train mse={train_metrics['mse']:.6f} mae={train_metrics['mae']:.6f} | "
                    f"val mse={val_metrics['mse']:.6f} mae={val_metrics['mae']:.6f} | "
                    f"test mse={test_metrics['mse']:.6f} mae={test_metrics['mae']:.6f}"
                )
            else:
                print(
                    f"[Epoch {epoch + 1}] train mse={train_metrics['mse']:.6f} mae={train_metrics['mae']:.6f} | "
                    f"val mse={val_metrics['mse']:.6f} mae={val_metrics['mae']:.6f}"
                )
            if val_metrics["loss"] < best_monitor:
                best_monitor = float(val_metrics["loss"])
                best_epoch = epoch + 1
                best_state = self._capture_checkpoint_state(
                    best_epoch=best_epoch, best_monitor=best_monitor
                )
                bad_epochs = 0
            else:
                bad_epochs += 1
            self._write_metrics_latest(
                latest_path=latest_path,
                metrics=metrics,
                best_epoch=max(best_epoch, 0),
                best_monitor=best_monitor
                if best_monitor < float("inf")
                else float(val_metrics["loss"]),
                status="running",
                current_epoch=epoch + 1,
            )
            if scheduler is not None:
                scheduler.step()
            if patience > 0 and bad_epochs >= patience:
                print(f"[EarlyStop] best_epoch={best_epoch} best_val_loss={best_monitor:.6f}")
                self._write_metrics_latest(
                    latest_path=latest_path,
                    metrics=metrics,
                    best_epoch=max(best_epoch, 0),
                    best_monitor=best_monitor
                    if best_monitor < float("inf")
                    else float(val_metrics["loss"]),
                    status="early_stop_pending_test",
                    current_epoch=epoch + 1,
                )
                break

        total_sec = time.perf_counter() - start_time
        print(f"[Total] train+eval total time: {total_sec:.1f}s ({total_sec / 60.0:.2f} min)")
        if best_state is not None:
            self._restore_checkpoint_state(best_state)
            print(
                f"[Best] loaded best weights from epoch {best_epoch} (val_loss={best_monitor:.6f})"
            )
        if best_state is not None and bool(getattr(self.args, "save_checkpoint", False)):
            ckpt_path, ckpt_latest_path = self._build_checkpoint_paths()
            self.save_checkpoint(
                ckpt_path, best_epoch=best_epoch, best_monitor=best_monitor, state=best_state
            )
            self.save_checkpoint(
                ckpt_latest_path, best_epoch=best_epoch, best_monitor=best_monitor, state=best_state
            )

        test_metrics = self.test()
        metrics["test"] = [test_metrics]
        run_info = {
            "best_epoch": int(best_epoch),
            "best_val_mse": float(best_monitor),
            "final_test_mse": float(test_metrics["mse"]),
            "monitor_target": "val",
            "args": self._collect_run_config(),
        }
        with open(save_path, "w", encoding="utf-8") as f:
            metrics["run_info"] = run_info
            json.dump(metrics, f, indent=2)
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Output] metrics saved to: {save_path}")
        print(f"[Output] latest metrics: {latest_path}")
        return self.model

    @torch.no_grad()
    def test(self):
        _test_data, test_loader = self._get_data(flag="test")
        self.model.ts_mlp.eval()
        self.model.pred_head.eval()
        self.model.eval()
        self.model.backbone.eval()

        total_sq = 0.0
        total_abs = 0.0
        total_count = 0

        max_batches = self.args.max_eval_batches
        for batch_idx, batch in enumerate(test_loader):
            x, y, img_grids, vid_grids, img_tokens, img_token_mask, vid_tokens, vid_token_mask = (
                self._unpack_batch(batch)
            )

            preds = self.model(
                x=x,
                img_grids=img_grids,
                vid_grids=vid_grids,
                img_tokens=img_tokens,
                img_token_mask=img_token_mask,
                vid_tokens=vid_tokens,
                vid_token_mask=vid_token_mask,
            )
            y_dev = y.to(preds.device)

            preds_eval, y_eval = self._maybe_denorm(preds, y_dev)
            diff = preds_eval - y_eval
            total_sq += float(torch.sum(diff**2).item())
            total_abs += float(torch.sum(torch.abs(diff)).item())
            total_count += int(y_eval.numel())

            if max_batches is not None and max_batches > 0 and (batch_idx + 1) >= max_batches:
                break

        test_metrics = {
            "mse": total_sq / max(1, total_count),
            "mae": total_abs / max(1, total_count),
            "loss": total_sq / max(1, total_count),
        }
        print(f"[Test] mse={test_metrics['mse']:.6f} mae={test_metrics['mae']:.6f}")
        if bool(getattr(self.args, "print_per_var_metrics", False)):
            per_var = self._per_var_metrics(test_loader)
            print(f"[Test] per-var mse={per_var['mse']} mae={per_var['mae']}")
        return test_metrics

    def _maybe_denorm(self, preds: torch.Tensor, y: torch.Tensor):
        if not bool(getattr(self.args, "eval_denorm", False)):
            return preds, y
        if self.data_mean is None or self.data_std is None:
            return preds, y
        mean = self.data_mean.to(preds.device).view(1, 1, -1)
        std = self.data_std.to(preds.device).view(1, 1, -1)
        return preds * std + mean, y * std + mean

    @torch.no_grad()
    def _per_var_metrics(self, loader):
        self.model.ts_mlp.eval()
        self.model.pred_head.eval()
        self.model.eval()
        self.model.backbone.eval()
        total_sq = None
        total_abs = None
        total_count = 0
        max_batches = self.args.max_eval_batches
        for batch_idx, batch in enumerate(loader):
            x, y, img_grids, vid_grids, img_tokens, img_token_mask, vid_tokens, vid_token_mask = (
                self._unpack_batch(batch)
            )
            preds = self.model(
                x=x,
                img_grids=img_grids,
                vid_grids=vid_grids,
                img_tokens=img_tokens,
                img_token_mask=img_token_mask,
                vid_tokens=vid_tokens,
                vid_token_mask=vid_token_mask,
            )
            y_dev = y.to(preds.device)
            preds_eval, y_eval = self._maybe_denorm(preds, y_dev)
            diff = preds_eval - y_eval
            sq = torch.sum(diff**2, dim=(0, 1))
            ab = torch.sum(torch.abs(diff), dim=(0, 1))
            if total_sq is None:
                total_sq = sq.detach().cpu()
                total_abs = ab.detach().cpu()
            else:
                total_sq += sq.detach().cpu()
                total_abs += ab.detach().cpu()
            total_count += int(y_eval.shape[0] * y_eval.shape[1])
            if max_batches is not None and max_batches > 0 and (batch_idx + 1) >= max_batches:
                break
        if total_sq is None:
            return {"mse": [], "mae": []}
        mse = (total_sq / max(1, total_count)).tolist()
        mae = (total_abs / max(1, total_count)).tolist()
        return {"mse": mse, "mae": mae}
