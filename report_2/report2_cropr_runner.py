"""Cropr experiments with Report 1 dataset, evaluation and measurement conventions.

All outputs stay under report_2. Importing does not start training.
"""


try:
    from . import cropr as cropr_module
except ImportError:
    import cropr as cropr_module


import csv


import gc


import hashlib


import importlib.metadata


import json


import math


import types


import os


import platform


import random


import shutil


import time


import threading


import traceback


from concurrent.futures import ThreadPoolExecutor


from collections import deque


from collections import Counter


from contextlib import contextmanager


from datetime import datetime


from pathlib import Path


import numpy as np


import torch


from PIL import Image


from torch import nn


from torch.nn import functional as F


from torch.profiler import ProfilerActivity, profile, record_function


from torch.utils.data import DataLoader, Dataset, default_collate


from tqdm.auto import tqdm


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    # Top-level, importable function for Windows spawn.
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _tf32_settings(enabled):
    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    if hasattr(matmul, "fp32_precision"):
        targets = (matmul, cudnn.conv, cudnn.rnn)
        key, value = "fp32_precision", "tf32" if enabled else "ieee"
    else:
        targets = (matmul, cudnn)
        key, value = "allow_tf32", bool(enabled)
    return targets, key, value


def _set_tf32(enabled):
    targets, key, value = _tf32_settings(enabled)
    for obj in targets:
        setattr(obj, key, value)


@contextmanager
def evaluation_tf32(enabled):
    """Scope TF32 to evaluation; preserve the caller's precision API and flags."""
    targets, key, value = _tf32_settings(enabled)
    previous = [getattr(obj, key) for obj in targets]
    try:
        for obj in targets:
            setattr(obj, key, value)
        yield
    finally:
        for obj, saved in zip(targets, previous):
            setattr(obj, key, saved)


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(obj, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def _state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(str((name, tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_csv(path, rows):
    if rows:
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with open(temporary, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class SavedSplitDataset(Dataset):
    """All worker dependencies are instance data, not notebook globals."""
    def __init__(self, task, part, entries, root, size, class_to_idx, val_labels):
        self.task, self.part = task, part
        self.entries, self.root, self.size = entries, Path(root), size
        self.class_to_idx, self.val_labels = class_to_idx, val_labels

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return self.get_item(index, self.part == "train" and random.random() < 0.5)

    def get_item(self, index, flip):
        image, target = self.decode_item(index, flip)
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float().div_(255)
        if self.task != "classification":
            target = torch.from_numpy(target).long()
        return image, target

    def decode_item(self, index, flip):
        """Pillow/NumPy only: safe to call without entering Torch in a worker."""
        # Explicit flip avoids sharing random state between loading threads.
        entry = self.entries[index]
        image_path = entry if self.task == "classification" else entry["image"]
        with Image.open(self.root / image_path) as im:
            image = im.convert("RGB").resize(
                (self.size, self.size), Image.Resampling.BICUBIC
            )
        if flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = np.array(image, dtype=np.uint8, copy=True)
        if self.task == "classification":
            label = (self.val_labels[Path(entry).stem] if self.part == "val"
                     else self.class_to_idx[Path(entry).parent.name])
            self._report_loading(index)
            return image, label
        with Image.open(self.root / entry["mask"]) as im:
            mask = im.resize((self.size, self.size), Image.Resampling.NEAREST)
        if flip:
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        target = np.array(mask, dtype=np.int64, copy=True)
        if target.ndim != 2 or target.min() < 0 or target.max() > 150:
            raise ValueError(f"Unexpected ADE20K mask: {entry['mask']}")
        target = np.where(target == 0, 255, target - 1)
        self._report_loading(index)
        return image, target

    def _report_loading(self, index):
        # Installed only during a sequential dev probe in this process.
        progress = getattr(self, "_loading_progress", None)
        if progress is not None and index < progress.total:
            progress.update(1)


@contextmanager
def first_batch_progress(loader):
    """Real image counts for threads/zero workers; batch count for processes."""
    per_image = isinstance(loader, ThreadedImageLoader) or loader.num_workers == 0
    total = min(loader.batch_size, len(loader.dataset)) if per_image else 1
    # Plain tqdm renders reliably in notebook text output as well as terminals.
    from tqdm import tqdm as text_progress
    with text_progress(total=total, desc="First dev batch: images decoded" if per_image
                       else "First dev batch: waiting for workers", unit="image" if per_image
                       else "batch", mininterval=0.1, leave=True) as progress:
        if per_image:
            loader.dataset._loading_progress = progress
        try:
            yield progress, per_image
        finally:
            if per_image:
                del loader.dataset._loading_progress


@contextmanager
def loader_timer(label, interval=5.0):
    """Report wall time, including while blocked; diagnostic, not a timeout."""
    started = time.perf_counter()
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(interval):
            tqdm.write(f"{label}: still waiting ({time.perf_counter() - started:.1f}s)")

    print(f"{label}...", flush=True)
    reporter = threading.Thread(target=heartbeat, name="loader-timer", daemon=True)
    reporter.start()
    status = "finished"
    try:
        yield
    except BaseException:
        status = "interrupted/failed"
        raise
    finally:
        stopped.set()
        reporter.join()
        print(f"{label}: {status} in {time.perf_counter() - started:.2f}s", flush=True)


def timed_image_batches(loader, label):
    """Measure consumer wait for CPU batches, not GPU time or total worker work.

    Prefetch overlaps training. Update one bar without adding CUDA synchronization.
    """
    iterator = iter(loader)
    batches = 0
    total = maximum = 0.0
    progress = tqdm(total=len(loader), desc=label, unit="batch", mininterval=1.0)
    try:
        while True:
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            elapsed = time.perf_counter() - started
            batches += 1
            total += elapsed
            maximum = max(maximum, elapsed)
            progress.set_postfix_str(f"load mean={total / batches:.3f}s", refresh=False)
            yield batch
            progress.update(1)
    finally:
        if hasattr(iterator, "close"):
            iterator.close()
        progress.close()


class ThreadedImageLoader:
    """Bounded, ordered CPU image prefetch without Windows child interpreters.

    NUM_WORKERS counts threads for this backend. Each pending task is one batch;
    at most num_workers batches are prefetched. Random choices happen in the
    iterator thread, so completion order cannot change shuffle or augmentation.
    """
    def __init__(self, dataset, batch_size, num_workers, shuffle, seed, pin_memory=True):
        self.dataset, self.batch_size = dataset, batch_size
        self.num_workers, self.shuffle = num_workers, shuffle
        self.generator = torch.Generator().manual_seed(seed)
        self.pin_memory = pin_memory

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _batch(self, choices):
        samples = []
        for index, flip in choices:
            try:
                samples.append(self.dataset.decode_item(index, flip))
            except Exception as exc:
                raise RuntimeError(
                    f"Image loading failed: {self.dataset.task}/{self.dataset.part} "
                    f"entry={self.dataset.entries[index]!r}: {exc}") from exc
        return samples

    def __iter__(self):
        indices = (torch.randperm(len(self.dataset), generator=self.generator).tolist()
                   if self.shuffle else list(range(len(self.dataset))))
        flips = (torch.rand(len(indices), generator=self.generator).lt(0.5).tolist()
                 if self.dataset.part == "train" else [False] * len(indices))
        batches = iter([list(zip(indices[i:i+self.batch_size], flips[i:i+self.batch_size]))
                        for i in range(0, len(indices), self.batch_size)])
        pool = ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix="image-loader")
        pending = deque()
        try:
            for _ in range(self.num_workers):
                choices = next(batches, None)
                if choices is not None:
                    pending.append(pool.submit(self._batch, choices))
            while pending:
                samples = pending.popleft().result()
                choices = next(batches, None)
                if choices is not None:
                    pending.append(pool.submit(self._batch, choices))
                # All Torch conversion, collation and pinning stay in the consumer.
                images, targets = zip(*samples)
                images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous().float().div_(255)
                targets = (torch.tensor(targets, dtype=torch.long)
                           if self.dataset.task == "classification" else
                           torch.from_numpy(np.stack(targets)).long())
                batch = [images, targets]
                if self.pin_memory:
                    batch = [tensor.pin_memory() for tensor in batch]
                yield batch
        finally:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)


def loss_sum_and_count(logits, target, task):
    if task == "classification":
        return F.cross_entropy(logits, target, reduction="sum"), target.numel()
    count = int((target != 255).sum().item())
    if count == 0:
        return logits.sum() * 0, 0
    return F.cross_entropy(logits, target, ignore_index=255, reduction="sum"), count


@contextmanager
def component_labels(model, task):
    originals, selected = [], []
    matched = {"attention": [], "mlp": [], "head": []}
    for name, module in model.backbone.named_modules():
        leaf = name.rsplit(".", 1)[-1].lower()
        kind = ("attention" if leaf in {"attn", "self_attn", "attention"}
                else "mlp" if leaf in {"mlp", "ffn"} else None)
        if kind and not any(name.startswith(parent + ".") for parent in selected):
            selected.append(name)
            matched[kind].append(name)
            originals.append((module, module.forward, kind))
    head = getattr(model, f"{task}_head")
    originals.append((head, head.forward, "head"))
    matched["head"].append(f"{task}_head")

    def wrapped(original, label):
        def forward(*args, **kwargs):
            with record_function(f"component/{label}"):
                return original(*args, **kwargs)
        return forward

    try:
        for module, original, kind in originals:
            module.forward = wrapped(original, kind)
        yield matched
    finally:
        for module, original, _ in originals:
            module.forward = original


_ATTENTION_LAYOUT_OPS = {
    # Embedding gathers table rows; no matrix/conv arithmetic is performed.
    'embedding',
    'alias', 'as_strided', 'clone', 'contiguous', 'copy_', 'detach', 'empty',
    'empty_like', 'empty_strided', 'expand', 'flatten', 'index', 'index_select',
    'lift_fresh', 'new_empty', 'new_empty_strided', 'permute', 'reshape', 'select',
    'slice', 'split', 'split_with_sizes', 'squeeze', 'stack', 't', 'to',
    'transpose', 'unbind', 'unflatten', 'unsqueeze', 'view', '_unsafe_view',
    '_to_copy', 'cat', 'gather', 'scatter', 'scatter_', 'repeat',
    'repeat_interleave', 'narrow', 'zeros', 'zeros_like', 'ones', 'ones_like',
    'full', 'full_like', 'fill_', 'zero_', 'arange', 'scalar_tensor',
    '_local_scalar_dense', 'is_nonzero', 'resolve_conj', 'resolve_neg',
}


_ATTENTION_EXCLUDED_ARITHMETIC = {
    'add', 'add_', 'sub', 'sub_', 'mul', 'mul_', 'div', 'div_', 'neg', 'pow',
    'exp', 'log', 'sqrt', 'rsqrt', 'sin', 'cos', 'softmax', '_softmax',
    '_safe_softmax', 'log_softmax', '_log_softmax', 'native_layer_norm',
    'layer_norm', 'rms_norm', 'native_group_norm', 'group_norm', 'dropout',
    'native_dropout', 'bernoulli', 'bernoulli_', 'rand', 'rand_like', 'randn',
    'uniform_', 'normal_', 'masked_fill', 'masked_fill_', 'where', 'tril',
    'triu', 'eq', 'ne', 'gt', 'ge', 'lt', 'le', 'logical_not', 'logical_and',
    'logical_or', 'bitwise_not', 'bitwise_and', 'bitwise_or', 'sum', 'mean',
    'amax', 'amin', 'max', 'min', 'var', 'var_mean', 'clamp', 'clamp_min',
    'clamp_max', 'relu', 'gelu', 'sigmoid', 'silu', 'tanh', 'abs',
    'linalg_vector_norm', 'norm', 'topk', 'sort', 'argsort', 'argmax',
    'argmin', 'scatter_add', 'scatter_add_', 'scatter_reduce',
}


def _find_attention_modules(model, module_names=None):
    """Return non-overlapping attention roots, relative to model.backbone."""
    modules = dict(model.backbone.named_modules())
    if module_names is not None:
        names = list(module_names)
        if len(set(names)) != len(names) or any(n not in modules for n in names):
            raise ValueError('Attention module names must be unique existing backbone paths.')
        names.sort(key=lambda n: (n.count('.'), n))
        for i, name in enumerate(names):
            if any(p == '' or name.startswith(p + '.') for p in names[:i]):
                raise ValueError('Attention module paths must not overlap (would double count).')
    else:
        names = []
        for name, module in modules.items():
            leaf = name.rsplit('.', 1)[-1].lower()
            if (leaf in {'attn', 'self_attn', 'attention'} or
                    'attention' in type(module).__name__.lower()):
                if not any(p == '' or name.startswith(p + '.') for p in names):
                    names.append(name)
    if not names:
        raise ValueError('No attention modules found.  Supply module_names relative to model.backbone.')
    return {n: modules[n] for n in names}


def _sdpa_dense_flops(q_shape, k_shape, v_shape, *args, **kwargs):
    """Dense QK^T + AV only, BHND layout.  Masked attention is a dense estimate."""
    if not all(len(s) == 4 for s in (q_shape, k_shape, v_shape)):
        raise ValueError('Expected BHND SDPA tensors; refusing an ambiguous attention count.')
    b, h, nq, dq = q_shape
    bk, hk, nk, dk = k_shape
    bv, hv, nv, dv = v_shape
    if not (b == bk == bv and dq == dk and nk == nv and hk == hv and h % hk == 0):
        raise ValueError('Unexpected SDPA tensor dimensions.')
    return int(2 * b * h * nq * nk * (dq + dv))


def measure_forward_flops(model, images, task, directory=None,
                            module_names=None, custom_mapping=None):
    """Whole-model and attention counts from the same eval forward.

    Returns whole-model plus attention summary fields. Optional directory gets
    audited operator ledgers and a per-module CSV. Counts include projections inside
    selected attention roots and supported attention products, using 2 FLOPs/MAC.
    Softmax, bias additions, normalization, activations, masking, sorting and
    memory movement are excluded.  Thus this is NOT an all-instruction count.
    Masking a dense attention matrix does not reduce this count.  Sparse/custom
    kernels need explicit formulas, supplied through custom_mapping using the
    FlopCounterMode API; audit those formulas against the implemented algorithm.
    module_names is a list of paths relative to model.backbone, useful when new
    attention implementations use different names.  Code must run eagerly.
    """
    from torch.utils.flop_counter import FlopCounterMode
    from torch.utils._python_dispatch import TorchDispatchMode

    if hasattr(model, '_orig_mod') or hasattr(model.backbone, '_orig_mod'):
        raise ValueError('Count eager model operations; torch.compile can hide custom kernels.')
    if images.ndim != 4 or images.shape[0] < 1:
        raise ValueError('images must be a nonempty [B, C, H, W] tensor.')
    selected = _find_attention_modules(model, module_names)
    # Explicit formulas for standard BHND fused SDPA variants, independent of
    # profiler CUDA kernel attribution and of version-specific registry gaps.
    mapping = {}
    for name in ('_scaled_dot_product_flash_attention',
                 '_scaled_dot_product_flash_attention_for_cpu',
                 '_scaled_dot_product_efficient_attention',
                 '_scaled_dot_product_cudnn_attention'):
        if hasattr(torch.ops.aten, name):
            mapping[getattr(torch.ops.aten, name)] = _sdpa_dense_flops
    mapping.update(custom_mapping or {})
    counter = FlopCounterMode(display=False, custom_mapping=mapping)
    # Freeze the counted operator families across PyTorch versions.
    allowed = {'mm', 'addmm', 'bmm', 'baddbmm', '_scaled_mm',
               'convolution', '_convolution', 'cudnn_convolution',
               '_slow_conv2d_forward', 'convolution_overrideable'}
    counter.flop_registry = {
        op: formula for op, formula in counter.flop_registry.items()
        if op in mapping or (str(op).startswith('aten.') and
                             str(op).rsplit('.', 1)[-1] in allowed)
    }
    ledger = {name: {'calls': 0, 'flops': 0, 'by_operator': Counter(),
                     'observed_operators': Counter(), 'operator_input_shapes': {},
                     'input_shapes': []}
              for name in selected}
    active = []
    global_observed = Counter()

    class AuditMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            global_observed[str(func._overloadpacket)] += 1
            if active:
                entry = ledger[active[-1]]
                op = str(func._overloadpacket)
                entry['observed_operators'][op] += 1
                if func._overloadpacket in counter.flop_registry:
                    shapes = [list(x.shape) for x in args if isinstance(x, torch.Tensor)]
                    shapes += [list(x.shape) for x in (kwargs or {}).values()
                               if isinstance(x, torch.Tensor)]
                    signatures = entry['operator_input_shapes'].setdefault(op, [])
                    if shapes not in signatures:
                        signatures.append(shapes)
            return func(*args, **(kwargs or {}))

    def snapshot():
        return dict(counter.get_flop_counts().get('Global', {}))

    def wrap(original, name):
        def forward(*args, **kwargs):
            before = snapshot()
            entry = ledger[name]
            entry['calls'] += 1
            shapes = [list(x.shape) for x in args if isinstance(x, torch.Tensor)]
            shapes += [list(x.shape) for x in kwargs.values() if isinstance(x, torch.Tensor)]
            if shapes not in entry['input_shapes']:
                entry['input_shapes'].append(shapes)
            active.append(name)
            try:
                return original(*args, **kwargs)
            finally:
                active.pop()
                for op, value in snapshot().items():
                    delta = int(value - before.get(op, 0))
                    if delta:
                        entry['by_operator'][str(op)] += delta
                        entry['flops'] += delta
        return forward

    # Preserve exact per-submodule eval/train flags and original forward lookup.
    training_flags = [(m, m.training) for m in model.modules()]
    originals = []
    try:
        model.eval()
        for name, module in selected.items():
            originals.append((module, 'forward' in module.__dict__,
                              module.__dict__.get('forward')))
            module.forward = wrap(module.forward, name)
        with torch.inference_mode(False), torch.no_grad(), counter, AuditMode():
            output = model(images, task)
            del output
    finally:
        for module, had_instance_forward, old in reversed(originals):
            if had_instance_forward:
                module.forward = old
            else:
                delattr(module, 'forward')
        for module, flag in training_flags:
            module.training = flag

    registered = {str(k) for k in counter.flop_registry}
    unknown = Counter()
    excluded = Counter()
    layout = Counter()
    rows = []
    for name, entry in ledger.items():
        for op, calls in entry['observed_operators'].items():
            leaf = op.rsplit('.', 1)[-1]
            if op in registered or op in entry['by_operator']:
                continue
            if op.startswith('aten.') and leaf in _ATTENTION_LAYOUT_OPS:
                layout[op] += calls
            elif op.startswith('aten.') and leaf in _ATTENTION_EXCLUDED_ARITHMETIC:
                excluded[op] += calls
            else:
                unknown[op] += calls
        rows.append({
            'module': name, 'calls': entry['calls'],
            'counted_forward_flops_per_image': entry['flops'] / images.shape[0],
            'counted_forward_gflops_per_image': entry['flops'] / images.shape[0] / 1e9,
            'parameters': sum(p.numel() for p in selected[name].parameters()),
            'trainable_parameters': sum(p.numel() for p in selected[name].parameters() if p.requires_grad),
        })
    missed = [n for n, e in ledger.items() if not e['calls'] or not e['flops']]
    coverage_ok = not unknown and not missed
    unique_parameters = {id(p): p for m in selected.values() for p in m.parameters()}
    counted = sum(e['flops'] for e in ledger.values()) / images.shape[0] / 1e9
    global_counts = {str(op): int(v) for op, v in snapshot().items()}
    global_unknown, global_excluded, global_layout = Counter(), Counter(), Counter()
    # Conventional matrix/conv FLOPs deliberately omit interpolation/pooling,
    # scalar arithmetic, normalization and data movement.  Record all omissions.
    model_exclusions = _ATTENTION_EXCLUDED_ARITHMETIC | {
        'upsample_bilinear2d', 'upsample_bicubic2d', 'upsample_nearest2d',
        '_upsample_bilinear2d_aa', '_upsample_bicubic2d_aa',
        'adaptive_avg_pool2d', '_adaptive_avg_pool2d', 'avg_pool2d',
        'max_pool2d_with_indices', 'constant_pad_nd', 'reflection_pad2d',
        'replication_pad2d', 'pad', 'batch_norm', 'native_batch_norm',
        '_native_batch_norm_legit_no_training', 'leaky_relu', 'hardswish',
    }
    for op, calls in global_observed.items():
        leaf = op.rsplit('.', 1)[-1]
        if op in registered or op in global_counts:
            continue
        if op.startswith('aten.') and leaf in _ATTENTION_LAYOUT_OPS:
            global_layout[op] += calls
        elif op.startswith('aten.') and leaf in model_exclusions:
            global_excluded[op] += calls
        else:
            global_unknown[op] += calls
    whole = sum(global_counts.values()) / images.shape[0] / 1e9
    whole_ok = not global_unknown and whole > 0
    if counted > whole + 1e-8:
        raise RuntimeError('Attention count exceeds whole-model count; possible overlapping scopes.')
    summary = {
        'whole_model_forward_gflops_per_image': whole if whole_ok else None,
        'whole_model_forward_gmacs_per_image': whole / 2 if whole_ok else None,
        'whole_model_counted_partial_gflops_per_image': whole,
        'whole_model_flops_coverage_ok': whole_ok,
        'whole_model_flops_unmapped_ops': ', '.join(sorted(global_unknown)),
        'whole_model_flops_convention': 'dense_matmul_conv_2_flops_per_mac',
        'whole_model_flops_counter_version': 3,
        'attention_forward_gflops_per_image': counted if coverage_ok else None,
        'attention_counted_partial_gflops_per_image': counted,
        'attention_flops_coverage_ok': coverage_ok,
        'attention_flops_convention': 'dense_matmul_2_flops_per_mac',
        'attention_flops_counter_version': 3,
        'attention_flops_unmapped_ops': ', '.join(sorted(unknown)),
        'attention_flops_uncounted_modules': ', '.join(missed),
        'attention_module_count': len(selected),
        'attention_parameters': sum(p.numel() for p in unique_parameters.values()),
        'attention_trainable_parameters': sum(p.numel() for p in unique_parameters.values() if p.requires_grad),
    }
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'whole_model_flops.json', {
            **summary, 'input_shape': list(images.shape), 'task': task,
            'torch_version': torch.__version__, 'passes': 1,
            'scope': 'Complete model(images, task), including active task head.',
            'by_operator_flops_per_batch': global_counts,
            'observed_operators': global_observed,
            'unmapped_operators': global_unknown,
            'excluded_arithmetic_operators': global_excluded,
            'layout_and_memory_operators': global_layout,
            'notes': [
                '2 FLOPs per MAC; counts matrix products, convolutions and dense SDPA products.',
                'Includes supported scorer/merge/reconstruction products anywhere in the model.',
                'Excludes softmax, normalization, activation, interpolation, pooling, sorting and elementwise arithmetic.',
                'Coverage means coverage of this convention, not every arithmetic instruction.',
                'Masks do not imply sparse execution; custom kernels require audited custom_mapping formulas.',
                'GMACs = GFLOPs / 2 under this convention; check each paper before comparing.',
                'Only executed paths are audited. Opaque external CUDA calls require manual review.',
            ],
        })
        write_json(directory / 'attention_flops.json', {
            **summary, 'torch_version': torch.__version__,
            'input_shape': list(images.shape), 'task': task,
            'passes': 1, 'mode': 'eval_forward_only',
            'scope': 'Matched attention modules in backbone; includes their internal projections.',
            'matched_modules': list(selected), 'by_module': ledger,
            'unmapped_operators': unknown, 'excluded_arithmetic_operators': excluded,
            'layout_and_memory_operators': layout,
            'custom_formula_operators': [str(k) for k in mapping],
            'notes': [
                'One multiply-add counts as 2 FLOPs.  Values are per image.',
                'Includes registered dense matrix/conv products and SDPA QK^T + AV.',
                'Excludes softmax, bias, normalization, activations, masking, sorting and data movement.',
                'coverage_ok only refers to this declared convention in the matched scopes.',
                'Unknown kernels or unexecuted modules make the comparable summary null; inspect the ledger.',
                'Masked SDPA is counted as dense work; zeros or masks alone do not imply sparse execution.',
                'Compression/routing outside matched attention modules is not included.',
                'For dynamic token counts, sample multiple real images and report distribution statistics.',
                'This does not measure attention backward or optimizer work.',
                'Counts are algorithmic estimates, not hardware instruction or time measurements.',
            ],
        })
        write_csv(directory / 'attention_flops_by_module.csv', rows)
    return summary


def measure_attention_flops(*args, **kwargs):
    """Compatibility alias; now also returns whole-model forward counts."""
    return measure_forward_flops(*args, **kwargs)


def measure_flops_on_dataset(model, dataset, task, batch_size, batches, seed,
                             directory, module_names=None, custom_mapping=None):
    """Deterministic dev subset; reports mean and variation of batch averages."""
    n = min(len(dataset), batch_size * batches)
    indices = np.random.default_rng(seed).choice(len(dataset), size=n, replace=False)
    device = next(model.parameters()).device
    directory = Path(directory)
    samples = []
    weights = []
    for offset in range(0, n, batch_size):
        chosen = indices[offset:offset + batch_size]
        images = torch.stack([dataset[int(i)][0] for i in chosen]).to(device)
        sample = measure_forward_flops(
            model, images, task, directory / 'flop_samples' / f'batch_{len(samples):03d}',
            module_names=module_names, custom_mapping=custom_mapping)
        samples.append(sample)
        weights.append(len(chosen))
        del images
    summary = dict(samples[0])
    for prefix in ('whole_model', 'attention'):
        ok = all(row[f'{prefix}_flops_coverage_ok'] for row in samples)
        summary[f'{prefix}_flops_coverage_ok'] = ok
        key = f'{prefix}_forward_gflops_per_image'
        partial = f'{prefix}_counted_partial_gflops_per_image'
        vals = np.array([row[partial] for row in samples])
        mean = float(np.average(vals, weights=weights))
        summary[partial] = mean
        summary[key] = mean if ok else None
        summary[f'{prefix}_forward_gflops_batch_mean_std'] = (
            float(vals.std(ddof=1)) if ok and len(vals) > 1 else (0.0 if ok else None))
        summary[f'{prefix}_forward_gflops_batch_mean_min'] = float(vals.min()) if ok else None
        summary[f'{prefix}_forward_gflops_batch_mean_max'] = float(vals.max()) if ok else None
        summary[f'{prefix}_flops_unmapped_ops'] = ', '.join(sorted({
            op for row in samples for op in row[f'{prefix}_flops_unmapped_ops'].split(', ') if op}))
    summary['attention_flops_uncounted_modules'] = ', '.join(sorted({
        op for row in samples for op in row['attention_flops_uncounted_modules'].split(', ') if op}))
    total = summary['whole_model_forward_gflops_per_image']
    summary['whole_model_forward_gmacs_per_image'] = total / 2 if total is not None else None
    summary.update(flops_sample_images=n, flops_sample_batches=len(samples), flops_sample_seed=seed)
    write_json(directory / 'forward_flops.json', {
        **summary, 'dataset_indices': indices.tolist(), 'batch_sizes': weights,
        'samples': samples,
        'notes': ['Std/min/max describe batch-average GFLOPs/image, not per-image variation.',
                  'Raw operators, exclusions and unknowns are recorded in flop_samples/batch_*/.',
                  'No legacy profiler partial estimate is used as a replacement for missing coverage.'],
    })
    return summary


def _component_time_summary(prof, matched):
    """Partition exclusive CPU-attributed CUDA time by enclosing component.

    Count CPU-side self device time only, never GPU events as well: adding both
    would double-count the same kernels. Missing attribution remains unavailable.
    """
    durations = dict.fromkeys(("attention", "mlp", "head", "other"), 0.0)
    negative_events = 0
    for event in prof.events():
        if str(getattr(event, "device_type", "")).split(".")[-1] != "CPU":
            continue
        scope = event
        component = None
        inside = False
        while scope is not None:
            key = getattr(scope, "name", "")
            if component is None and key in {"component/attention", "component/mlp", "component/head"}:
                component = key.split("/")[-1]
            if key == "component/total":
                inside = True
                break
            scope = getattr(scope, "cpu_parent", None)
        if not inside:
            continue
        value = getattr(event, "self_device_time_total", None)
        if value is None:
            value = getattr(event, "self_cuda_time_total", 0.0)
        value = float(value)
        if value < -1e-6:
            negative_events += 1
        durations[component or "other"] += max(value, 0.0)
    total = sum(durations.values())
    reasons = []
    if total <= 0:
        reasons.append("No CUDA time attributed to CPU operations inside the forward scope.")
    if negative_events:
        reasons.append("Negative exclusive CUDA times; profiler attribution is inconsistent.")
    for kind in ("attention", "mlp", "head"):
        if not matched[kind]:
            reasons.append(f"No matched {kind} modules.")
        elif durations[kind] <= 0:
            reasons.append(f"No CUDA time attributed to matched {kind} modules.")
    ok = not reasons
    return {
        "component_gpu_attribution_available": ok,
        "component_timing_method": "profiler_exclusive_cuda_time_by_cpu_scope_v2",
        "component_gpu_attribution_reason": "; ".join(reasons),
        "component_total_gpu_us": total,
        **{f"{k}_gpu_time_us": v for k, v in durations.items()},
        **{f"{k}_gpu_time_pct": 100 * v / total if ok else None
           for k, v in durations.items()},
    }


class _Runner:


    def __init__(self, model_factory, config):
        self.model_factory = model_factory
        self.c = dict(config)
        self.c.setdefault("TRAIN_PRECISION", "fp32")
        self.c.setdefault("LOADER_BACKEND", "process")
        self.c.setdefault("FAIL_FAST", True)
        self.device = torch.device("cuda:0")
        self.root = config["ROOT"]
        self.roots = {"classification": config["IMAGENET_ROOT"],
                      "segmentation": config["ADE_ROOT"]}
        self.split_paths = {
            "classification": config["SPLIT_ROOT"] / "imagenet_split.json",
            "segmentation": config["SPLIT_ROOT"] / "ade20k_split.json",
        }


    def prepare(self):
        c = self.c
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable.  Select your CUDA PyTorch notebook kernel.")
        if c["STRICT_DETERMINISM"]:
            if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
                raise RuntimeError("Strict determinism requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization.")
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(c["STRICT_DETERMINISM"])
        torch.cuda.set_device(self.device)
        if c.get("TRAIN_PRECISION", "fp32") == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is not supported by this GPU/PyTorch build. Set TRAIN_PRECISION='fp32'.")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        _set_tf32(False)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self.splits = {task: json.loads(path.read_text(encoding="utf-8"))
                       for task, path in self.split_paths.items()}
        self.class_to_idx = self.splits["classification"]["class_to_idx"]
        if (len(self.class_to_idx) != 1000 or
                set(self.class_to_idx.values()) != set(range(1000))):
            raise ValueError("ImageNet class_to_idx must map 1000 classes to 0..999.")
        self.val_labels = {}
        with open(c["IMAGENET_ROOT"] / "LOC_val_solution.csv", newline="",
                  encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                tokens = row["PredictionString"].split()
                image_id = row["ImageId"]
                if not tokens or len(tokens) % 5:
                    raise ValueError(f"Malformed localization label: {image_id}")
                synsets = set(tokens[::5])
                if len(synsets) != 1 or not synsets.issubset(self.class_to_idx):
                    raise ValueError(f"Ambiguous/unknown classification label: {image_id}: {synsets}")
                if image_id in self.val_labels:
                    raise ValueError(f"Duplicate validation label: {image_id}")
                self.val_labels[image_id] = self.class_to_idx[next(iter(synsets))]
        for task, split in self.splits.items():
            seen = set()
            for part in ("train", "dev", "val"):
                entries = split[f"{part}_files"]
                if not entries:
                    raise ValueError(f"Empty {task}/{part} split")
                for entry in tqdm(entries, desc=f"Validate {task}/{part}", unit="files"):
                    rel = entry if task == "classification" else entry["image"]
                    canonical = os.path.normcase(str((self.roots[task] / rel).resolve()))
                    if canonical in seen:
                        raise ValueError(f"Duplicate or overlapping split entry: {rel}")
                    seen.add(canonical)
                    paths = [self.roots[task] / rel]
                    if task == "segmentation":
                        paths.append(self.roots[task] / entry["mask"])
                    for path in paths:
                        if not path.is_file():
                            raise FileNotFoundError(path)
                    if task == "classification":
                        if part == "val" and Path(rel).stem not in self.val_labels:
                            raise ValueError(f"Missing validation label: {rel}")
                        if part != "val" and Path(rel).parent.name not in self.class_to_idx:
                            raise ValueError(f"Unknown training class: {rel}")
            print(task, {p: len(split[f"{p}_files"]) for p in ("train", "dev", "val")})
        self.out = self.root / "report_2" / "cropr measurements" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.out.mkdir(parents=True, exist_ok=False)
        configuration = {
            "model_definition_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "baseline_models.py").read_bytes()).hexdigest(),
            "training_mode": "cropr", "seed": c["SEED"], "input_size": c["SIZE"],
            "batch_size": c["BATCH_SIZE"], "micro_batch_size": c["MICRO_BATCH_SIZE"],
            "grad_accum_steps": c["GRAD_ACCUM_STEPS"], "eval_batch_size": c["EVAL_BATCH_SIZE"],
            "effective_train_batch_size": c["MICRO_BATCH_SIZE"] * c["GRAD_ACCUM_STEPS"],
            "epochs": c["EPOCHS"], "learning_rates": c["LR"], "weight_decay": c["WEIGHT_DECAY"],
            "cropr_variant": c["VARIANT"], "pruning_rate": c["PRUNING_RATE"],
            "cropr_lr": c["CROPR_LR"], "aux_weight": c["AUX_WEIGHT"],
            "adaptation_mode": "lora_cropr",
            "lora": {"rank": c["LORA_R"], "alpha": c["LORA_ALPHA"], "lr": c["LORA_LR"],
                     "targets": c["LORA_TARGETS"], "dropout": 0.0},
            "inference_lora_merged": True,
            "num_heads": c["NUM_HEADS"], "strict_determinism": c["STRICT_DETERMINISM"],
            "max_grad_norm": c["MAX_GRAD_NORM"] if math.isfinite(c["MAX_GRAD_NORM"]) else None,
            "optimizer": "AdamW", "scheduler": "CosineAnnealingLR",
            "train_precision": c["TRAIN_PRECISION"], "evaluation_precision": "fp32",
            "evaluation_tf32": c["EVAL_TF32"], "benchmark_precision": "fp32; TF32 disabled",
            "training_seconds_scope": "RGB loading + main/auxiliary forward/backward + optimizer; excludes dev and audits",
            "accumulation_loss_normalization": "main and auxiliary losses each normalized by actual valid targets per window",
            "pretrained_parameters_frozen": True, "cache_enabled": False,
            "augmentation": f"resize to {c['SIZE']}x{c['SIZE']}; training horizontal flip p=0.5",
            "segmentation_evaluation": f"{c['SIZE']}x{c['SIZE']} masks, 150 classes, ignore label 255",
            "heldout_split": "official validation; used after dev-based selection",
            "split_sha256": {t: hashlib.sha256(p.read_bytes()).hexdigest() for t,p in self.split_paths.items()},
            "split_sizes": {t: {p: len(s[f'{p}_files']) for p in ('train','dev','val')} for t,s in self.splits.items()},
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cropr_sha256": hashlib.sha256(Path(cropr_module.__file__).read_bytes()).hexdigest(),
            "gpu": torch.cuda.get_device_name(self.device), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "python": platform.python_version(),
            **{p: package_version(p) for p in ("torchvision", "transformers", "timm")},
            "loader_backend": c["LOADER_BACKEND"], "num_workers": c["NUM_WORKERS"],
            "model_names": c["MODEL_NAMES"], "tasks": c["TASKS"],
            "bench_warmup": c["BENCH_WARMUP"], "bench_repeats": c["BENCH_REPEATS"],
            "bench_trials": c["BENCH_TRIALS"], "flop_batches": c["FLOP_BATCHES"],
            "profile_repeats": c["PROFILE_REPEATS"], "whole_model_flops_convention": "dense_matmul_conv_2_flops_per_mac",
            "flops_counter_version": 3, "attention_module_names": c["ATTENTION_MODULE_NAMES"],
        }
        write_json(self.out / "configuration.json", configuration)
        print("Results:", self.out)
        print("Cropr variant:", c["VARIANT"], "| precision:", c["TRAIN_PRECISION"])


    def make_loader(self, task, part):
        c = self.c
        dataset = SavedSplitDataset(
            task, part, self.splits[task][f"{part}_files"], self.roots[task],
            c["SIZE"], self.class_to_idx if task == "classification" else {},
            self.val_labels if task == "classification" and part == "val" else {},
        )
        batch_size = (c["MICRO_BATCH_SIZE"] if part == "train" else c["EVAL_BATCH_SIZE"]) if c["TRAINING_MODE"] == "cropr" else c["BATCH_SIZE"]
        if c["NUM_WORKERS"] and c.get("LOADER_BACKEND", "process") == "thread":
            return ThreadedImageLoader(dataset, batch_size, c["NUM_WORKERS"],
                                       part == "train", c["SEED"])
        worker_args = {"multiprocessing_context": "spawn", "prefetch_factor": 1} if c["NUM_WORKERS"] else {}
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=part == "train",
            num_workers=c["NUM_WORKERS"], pin_memory=True, drop_last=False,
            generator=torch.Generator().manual_seed(c["SEED"]),
            worker_init_fn=seed_worker, persistent_workers=False, **worker_args,
        )


    def training_autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                              enabled=self.c.get("TRAIN_PRECISION", "fp32") == "bf16")


    @torch.inference_mode()
    def evaluate(self, model, loader, task):
        model.eval()
        loss_total = count_total = correct1 = correct5 = 0
        confusion = torch.zeros((150, 150), dtype=torch.int64, device=self.device)
        for images, target in timed_image_batches(loader, f"Evaluate {task}"):
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            with evaluation_tf32(self.c.get("EVAL_TF32", False)):
                logits = model(images, task)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite evaluation logits.")
            loss_sum, count = loss_sum_and_count(logits, target, task)
            loss_total += loss_sum.item()
            count_total += count
            if task == "classification":
                predictions = logits.topk(5, dim=1).indices
                correct1 += (predictions[:, 0] == target).sum().item()
                correct5 += (predictions == target[:, None]).any(dim=1).sum().item()
            else:
                valid = target != 255
                prediction = logits.argmax(dim=1)
                encoded = target[valid] * 150 + prediction[valid]
                confusion += torch.bincount(encoded, minlength=150 * 150).reshape(150, 150)
        if count_total == 0:
            raise ValueError("Evaluation split has no valid labels.")
        result = {"loss": loss_total / count_total}
        if task == "classification":
            result.update(top1_pct=100 * correct1 / count_total, top5_pct=100 * correct5 / count_total)
        else:
            cm = confusion.double()
            intersection = cm.diag()
            union = cm.sum(0) + cm.sum(1) - intersection
            present = union > 0
            result.update(
                miou_pct=100 * (intersection[present] / union[present]).mean().item(),
                pixel_accuracy_pct=100 * intersection.sum().item() / cm.sum().item(),
                classes_with_nonzero_union=int(present.sum().item()),
            )
        return result


    @torch.inference_mode()
    def benchmark(self, model, images, task, directory=None):
        model.eval()
        result, trial_rows = {}, []
        for batch in dict.fromkeys((1, self.c["BATCH_SIZE"])):
            tag = f"b{batch}"
            x = images[:batch].contiguous()
            if len(x) != batch:
                raise ValueError('Benchmark input is smaller than requested batch.')
            gpu_trials, wall_trials, peaks, all_times = [], [], [], []
            for trial in range(self.c["BENCH_TRIALS"]):
                for _ in range(self.c["BENCH_WARMUP"]):
                    model(x, task)
                torch.cuda.synchronize()
                # Allocate and initialize events outside timed work.
                pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                         for _ in range(self.c["BENCH_REPEATS"])]
                for start, end in pairs:
                    start.record()
                    end.record()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                for start, end in pairs:
                    start.record()
                    model(x, task)
                    end.record()
                torch.cuda.synchronize()
                times = np.array([a.elapsed_time(b) for a, b in pairs])
                peak = torch.cuda.max_memory_allocated() / 2**30
                # Separate uninstrumented wall-clock loop includes host overhead.
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(self.c["BENCH_REPEATS"]):
                    model(x, task)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                gpu_rate = float(batch * 1000 / times.mean())
                wall_rate = batch * self.c["BENCH_REPEATS"] / seconds
                gpu_trials.append(gpu_rate)
                wall_trials.append(wall_rate)
                peaks.append(peak)
                all_times.extend(times.tolist())
                trial_rows.append(dict(batch_size=batch, trial=trial + 1,
                    gpu_images_per_second=gpu_rate, wall_images_per_second=wall_rate,
                    gpu_latency_median_ms=float(np.median(times)),
                    gpu_latency_p95_ms=float(np.percentile(times, 95)),
                    wall_mean_batch_ms=seconds * 1000 / self.c["BENCH_REPEATS"],
                    gpu_latency_samples_ms=times.tolist(), peak_allocated_gib=peak))
            result.update({
                f"{tag}_latency_median_ms": float(np.median(all_times)),
                f"{tag}_latency_p95_ms": float(np.percentile(all_times, 95)),
                f"{tag}_images_per_second": float(np.mean(gpu_trials)),
                f"{tag}_images_per_second_std": float(np.std(gpu_trials, ddof=1)) if len(gpu_trials)>1 else None,
                f"{tag}_wall_images_per_second": float(np.mean(wall_trials)),
                f"{tag}_wall_images_per_second_std": float(np.std(wall_trials, ddof=1)) if len(wall_trials)>1 else None,
                f"{tag}_peak_allocated_gib": max(peaks),
            })
        result['benchmark_trials'] = self.c['BENCH_TRIALS']
        if directory is not None:
            write_json(Path(directory) / 'benchmark_trials.json', {
                **result, 'trials': trial_rows,
                'notes': ['Same checkpoint and resident GPU inputs across timing trials; no retraining.',
                          'Excludes loading, preprocessing and host-to-device transfer.',
                          'Wall throughput includes Python/CPU launch overhead and final synchronization.',
                          'CUDA-event latency describes GPU-stream elapsed time, not end-to-end request latency.',
                          'Repeated trials use one representative batch. Dynamic input-dependent methods need broader input sampling.'],
            })
        return result


    def profile_model(self, model, images, target, task, directory, profile_backward=False):
        repeats = self.c["PROFILE_REPEATS"]
        model.eval()
        with component_labels(model, task) as matched:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True, with_flops=True) as prof:
                with torch.no_grad():
                    for _ in range(repeats):
                        with record_function("component/total"):
                            model(images, task)
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(directory / "inference_trace.json"))
        averages = prof.key_averages()
        result = _component_time_summary(prof, matched)
        result["forward_partial_gflops_per_image"] = sum(float(e.flops or 0) for e in averages) / repeats / images.shape[0] / 1e9
        write_json(directory / "profiling.json", {
            **result, "matched_modules": matched, "batch_size": images.shape[0],
            "notes": [
                "Percentages describe profiled forward GPU time, not FLOP shares.",
                "Attention includes QKV/output projections inside attention modules.",
                "Profiler overhead is excluded from benchmark speed measurements.",
                "FLOPs are partial supported-operation estimates, not total compute.",
                "Training trace includes Cropr auxiliary loss and task head backward.",
                "Inference benchmark excludes loading, preprocessing and host-to-device transfer.",
            ],
        })
        return result


    def run(self):
        self.prepare()
        results = []
        for task in self.c["TASKS"]:
            for name in self.c["MODEL_NAMES"]:
                print(f"\n{'=' * 65}\nStarting {task}: {name}\n{'=' * 65}")
                try:
                    result = self.run_experiment(name, task)
                except Exception as exc:
                    result = {"model": name, "task": task, "status": "failed", "error": str(exc)}
                    (self.out / f"{task}_{name}_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        print("CUDA OOM: reduce MICRO_BATCH_SIZE for training, EVAL_BATCH_SIZE for evaluation, "
                              "or BATCH_SIZE for FP32 benchmarks.  GRAD_ACCUM_STEPS preserves the desired "
                              "effective training batch.  No automatic retry or batch change was performed.")
                    print("FAILED:", exc)
                    print("Full traceback:", self.out / f"{task}_{name}_error.txt")
                    if self.c.get("FAIL_FAST", True):
                        write_csv(self.out / "cropr measurements.csv", results + [result])
                        raise
                results.append(result)
                write_csv(self.out / "cropr measurements.csv", results)
                gc.collect()
                torch.cuda.empty_cache()
        print("\nFinished.  Measurements saved to:")
        print(self.out / "cropr measurements.csv")
        for result in results:
            print(result["task"], result["model"], "→", result["status"])
        return results, self.out


def backward_cropr_batch(model, images, target, task, main_normalizer, aux_normalizer,
                         aux_weight, autocast):
    """Two independent normalizers: image/pixel labels and auxiliary grid labels."""
    with autocast():
        logits = model(images, task)
    main, count = loss_sum_and_count(logits.float(), target, task)
    aux, aux_count = model.auxiliary_loss_sums(target)
    if not torch.isfinite(logits).all() or not torch.isfinite(main) or not torch.isfinite(aux):
        raise RuntimeError("Non-finite Cropr logits/loss; no optimizer update")
    if count or aux_count:
        (main / main_normalizer + aux_weight * aux / aux_normalizer).backward()
    model.auxiliary_predictions = []
    return main.detach().item(), count, aux.detach().item(), aux_count


def finish_cropr_step(model, optimizer, main_normalizer, main_count,
                      aux_normalizer, aux_count, max_norm):
    """Normalize final/partial accumulation windows without bias from ignored pixels.

    This separation is valid because the backbone is frozen and the auxiliary
    input is detached: main loss updates head/LoRA; auxiliary updates pruners.
    """
    head = getattr(model, f"{model.task}_head")
    for parameters, denominator, count in (
        (head.parameters(), main_normalizer, main_count),
        ((p for p in model.backbone.parameters() if p.requires_grad), main_normalizer, main_count),
        (model.pruners.parameters(), aux_normalizer, aux_count),
    ):
        for p in parameters:
            if not count:
                p.grad = None  # no AdamW decay for an all-ignored branch
            elif p.grad is not None:
                p.grad.mul_(denominator / count)
    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not parameters:
        optimizer.zero_grad(set_to_none=True)
        return None
    if main_count:
        for name, module in cropr_module._lora_modules(model.backbone).items():
            if module.lora_A.grad is None or module.lora_B.grad is None:
                raise RuntimeError(f"Disconnected LoRA gradients: {name}")
    if aux_count and any(p.queries.grad is None for p in model.pruners.values()):
        raise RuntimeError("Disconnected Cropr queries; refusing silent head-only training")
    # Keep the baseline head/LoRA clipping independent of auxiliary gradients.
    main_parameters = list(head.parameters()) + [
        p for p in model.backbone.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(main_parameters, max_norm, error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_(model.pruners.parameters(), max_norm, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


class _CroprRunner(_Runner):
    def profile_training(self, model, images, target, task, directory):
        model.train()
        repeats = self.c['PROFILE_REPEATS']
        try:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True, with_flops=True) as prof:
                for _ in range(repeats):
                    model.zero_grad(set_to_none=True)
                    with record_function('training/forward_and_cropr_head_backward'):
                        with self.training_autocast():
                            logits = model(images, task)
                        main, count = loss_sum_and_count(logits.float(), target, task)
                        aux, aux_count = model.auxiliary_loss_sums(target)
                        if not count:
                            raise ValueError('No valid labels in training profile batch')
                        (main / count + self.c['AUX_WEIGHT'] * aux / max(aux_count, 1)).backward()
                        model.auxiliary_predictions = []
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(directory / 'training_trace.json'))
            return {'forward_backward_partial_gflops_per_image':
                    sum(float(e.flops or 0) for e in prof.key_averages()) / repeats / len(images) / 1e9,
                    'training_profile_precision': self.c['TRAIN_PRECISION'],
                    'training_profile_batch_size': len(images)}
        finally:
            model.zero_grad(set_to_none=True)
            model.auxiliary_predictions = []
            model.eval()

    def run_experiment(self, name, task):
        c = self.c
        experiment_start = time.perf_counter()
        seed_everything(c['SEED'])
        gc.collect()
        torch.cuda.empty_cache()
        directory = self.out / f'{task}_{name}'
        directory.mkdir()
        loaders = {p: self.make_loader(task, p) for p in ('train', 'dev', 'val')}
        if len(loaders['dev'].dataset) < c['BATCH_SIZE']:
            raise ValueError('Dev split must contain a full benchmark batch')
        spec = dict(variant=c['VARIANT'], pruning_rate=c['PRUNING_RATE'],
                    input_size=c['SIZE'], num_heads=c['NUM_HEADS'])
        # Match Report 1's base/head construction before consuming any Cropr RNG.
        base = self.model_factory(name)
        # Cropr initialization must not shift the CPU/CUDA LoRA random stream.
        with torch.random.fork_rng():
            model = cropr_module.build_cropr_model(name, task, base=base, **spec)
        model = model.to(self.device)
        head = getattr(model, f'{task}_head')
        head.requires_grad_(True)
        base_hash = cropr_module._frozen_backbone_digest(model.backbone)
        model.eval()
        lora_spec = cropr_module.inject_lora(
            model, rank=c['LORA_R'], alpha=c['LORA_ALPHA'],
            targets=c['LORA_TARGETS'].get(name), unwrap_backbone_forward=False)
        if cropr_module._frozen_backbone_digest(model.backbone) != base_hash:
            raise RuntimeError('Pretrained weights changed during LoRA injection')
        initial_adapters = cropr_module._adapter_state(model.backbone)
        adapters = [p for p in model.backbone.parameters() if p.requires_grad]
        initial_queries = {k: p.queries.detach().cpu().clone() for k, p in model.pruners.items()}
        allowed = {id(p) for p in list(head.parameters()) + list(model.pruners.parameters()) + adapters}
        if {id(p) for p in model.parameters() if p.requires_grad} != allowed:
            raise RuntimeError('Unexpected trainable parameters outside the head/pruners/LoRA')
        # A real RGB dev image verifies shape, zero-pruning equivalence and query gradients.
        check_images, _ = next(iter(loaders['dev']))
        probe = check_images[:1].to(self.device)
        model.eval()
        saved_schedule = model.schedule.copy()
        try:
            model.schedule = dict.fromkeys(saved_schedule, 0)
            with torch.no_grad():
                expected = model.model(probe, task)
                actual = model(probe, task)
                originals = []
                try:
                    for path, module in cropr_module._lora_modules(model.backbone).items():
                        parent, _, leaf = path.rpartition('.')
                        owner = model.backbone.get_submodule(parent)
                        originals.append((owner, leaf, module))
                        setattr(owner, leaf, module.base)
                    original_logits = model.model(probe, task)
                    torch.testing.assert_close(expected, original_logits, rtol=0, atol=0)
                finally:
                    for owner, leaf, module in originals:
                        setattr(owner, leaf, module)
                del original_logits, originals
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            model.schedule = saved_schedule
        model.train()
        with self.training_autocast():
            actual = model(probe, task)
        shape = (1, 1000) if task == 'classification' else (1, 150, c['SIZE'], c['SIZE'])
        if tuple(actual.shape) != shape:
            raise ValueError(f'Unexpected Cropr output shape: {actual.shape}; expected {shape}')
        synthetic_target = torch.zeros((1,) if task == 'classification' else (1, c['SIZE'], c['SIZE']),
                                       device=self.device, dtype=torch.long)
        audit_loss = model.auxiliary_loss(synthetic_target)
        audit_loss.backward()
        for key, pruner in model.pruners.items():
            grad = pruner.queries.grad
            if grad is None or not torch.isfinite(grad).all() or not torch.count_nonzero(grad):
                raise RuntimeError(f'Query gradient audit failed at stage {key}')
        if any(p.grad is not None for p in model.backbone.parameters()):
            raise RuntimeError('Auxiliary loss must not update the backbone or LoRA adapters')
        actual.float().square().mean().backward()
        for path, module in cropr_module._lora_modules(model.backbone).items():
            for key in ('lora_A', 'lora_B'):
                grad = getattr(module, key).grad
                if grad is None or not torch.isfinite(grad).all():
                    raise RuntimeError(f'LoRA gradient audit failed: {path}.{key}')
        if not any(torch.count_nonzero(m.lora_B.grad).item()
                   for m in cropr_module._lora_modules(model.backbone).values()):
            raise RuntimeError('All LoRA B gradients are zero; refusing silent head-only training')
        if any(p.grad is not None for p in model.backbone.parameters() if not p.requires_grad):
            raise RuntimeError('Frozen pretrained parameters received gradients')
        model.zero_grad(set_to_none=True)
        model.auxiliary_predictions = []
        write_json(directory / 'cropr_setup.json', {
            **spec, 'schedule': model.schedule, 'prefix_tokens': model.prefix,
            'grid_size': model.grid_size, 'zero_pruning_equivalence': True,
            'query_gradient_connectivity': True, 'backbone_eval_mode': True,
            'lora': lora_spec, 'lora_initial_equivalence': True, 'lora_gradient_connectivity': True,
            'initial_state_hash_format': 'tensor_state_v1',
            'initial_head_sha256': _state_digest(head.state_dict()),
            'initial_lora_sha256': _state_digest(initial_adapters),
            'pretrained_backbone_sha256': base_hash,
        })
        del probe, actual, expected, synthetic_target, audit_loss, check_images
        optimizer = torch.optim.AdamW([
            {'params': head.parameters(), 'lr': c['LR'][task], 'name': 'head'},
            {'params': adapters, 'lr': c['LORA_LR'], 'name': 'lora'},
            {'params': model.pruners.parameters(), 'lr': c['CROPR_LR'], 'name': 'cropr'},
        ], weight_decay=c['WEIGHT_DECAY'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c['EPOCHS'][task])
        metric = 'top1_pct' if task == 'classification' else 'miou_pct'
        best_score, best_epoch = -float('inf'), None
        history = []
        total_steps = total_images = 0
        fit_start = time.perf_counter()
        for epoch in range(1, c['EPOCHS'][task] + 1):
            seed_everything(c['SEED'] + epoch)
            loaders['train'].generator.manual_seed(c['SEED'] + epoch)
            model.train()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            main_sum = aux_sum = 0.0
            main_count = aux_count = epoch_images = epoch_steps = 0
            window_main = window_aux = window_batches = 0
            max_grad_norm = 0.0
            main_den = aux_den = c['MICRO_BATCH_SIZE'] * c['GRAD_ACCUM_STEPS']
            if task == 'segmentation':
                main_den *= c['SIZE'] ** 2
                aux_den *= math.prod(model.grid_size)
            optimizer.zero_grad(set_to_none=True)
            for images, target in timed_image_batches(loaders['train'], f'{task}/{name} Cropr epoch {epoch}'):
                images = images.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                main, count, aux, auxiliary_count = backward_cropr_batch(
                    model, images, target, task, main_den, aux_den, c['AUX_WEIGHT'], self.training_autocast)
                main_sum += main
                aux_sum += aux
                main_count += count
                aux_count += auxiliary_count
                window_main += count
                window_aux += auxiliary_count
                window_batches += 1
                epoch_images += len(images)
                del images, target
                if window_batches == c['GRAD_ACCUM_STEPS']:
                    norm = finish_cropr_step(model, optimizer, main_den, window_main,
                                            aux_den, window_aux, c['MAX_GRAD_NORM'])
                    if norm is not None:
                        epoch_steps += 1
                        max_grad_norm = max(max_grad_norm, norm)
                    window_main = window_aux = window_batches = 0
            if window_batches:
                norm = finish_cropr_step(model, optimizer, main_den, window_main,
                                        aux_den, window_aux, c['MAX_GRAD_NORM'])
                if norm is not None:
                    epoch_steps += 1
                    max_grad_norm = max(max_grad_norm, norm)
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - started
            allocated = torch.cuda.max_memory_allocated() / 2**30
            reserved = torch.cuda.max_memory_reserved() / 2**30
            if not main_count or not aux_count or not epoch_steps:
                raise RuntimeError('No valid main/auxiliary targets or optimizer steps')
            if cropr_module._frozen_backbone_digest(model.backbone) != base_hash:
                raise RuntimeError('Frozen backbone changed during training')
            if not any(not torch.equal(p.queries.detach().cpu(), initial_queries[k])
                       for k, p in model.pruners.items()):
                raise RuntimeError('Queries did not change; refusing silent head-only training')
            current_adapters = cropr_module._adapter_state(model.backbone)
            if not any(not torch.equal(v, initial_adapters[k]) for k,v in current_adapters.items()):
                raise RuntimeError('LoRA adapters did not change; refusing silent head-only training')
            total_steps += epoch_steps
            total_images += epoch_images
            started = time.perf_counter()
            dev = self.evaluate(model, loaders['dev'], task)
            dev_seconds = time.perf_counter() - started
            row = dict(training_mode='cropr', train_precision=c['TRAIN_PRECISION'], epoch=epoch,
                       evaluation_precision='fp32', evaluation_tf32=c['EVAL_TF32'],
                       lr=optimizer.param_groups[0]['lr'], cropr_lr=optimizer.param_groups[2]['lr'],
                       lora_lr=optimizer.param_groups[1]['lr'], adaptation_mode='lora_cropr',
                       train_loss=main_sum / main_count, train_aux_loss=aux_sum / aux_count,
                       train_total_loss=main_sum / main_count + c['AUX_WEIGHT'] * aux_sum / aux_count,
                       train_seconds=train_seconds, train_images_per_second=epoch_images / train_seconds,
                       train_images=epoch_images, optimizer_steps=epoch_steps,
                       micro_batch_size=c['MICRO_BATCH_SIZE'], grad_accum_steps=c['GRAD_ACCUM_STEPS'],
                       effective_train_batch_size=c['MICRO_BATCH_SIZE'] * c['GRAD_ACCUM_STEPS'],
                       train_peak_allocated_gib=allocated, train_peak_reserved_gib=reserved,
                       max_preclip_grad_norm=max_grad_norm, dev_seconds=dev_seconds,
                       **{f'dev_{k}': v for k, v in dev.items()})
            history.append(row)
            write_csv(directory / 'epochs.csv', history)
            checkpoint = dict(
                model_name=name, task=task, epoch=epoch, cropr=spec,
                lora=lora_spec, lora_state_dict=current_adapters, adaptation_mode='lora_cropr',
                training_mode='cropr', train_precision=c['TRAIN_PRECISION'], evaluation_precision='fp32',
                evaluation_tf32=c['EVAL_TF32'], input_size=c['SIZE'], feature_dim=model.backbone.feature_dim,
                head_state_dict={k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                pruner_state_dict={k: v.detach().cpu().clone() for k, v in model.pruners.state_dict().items()},
                pretrained_backbone_sha256=base_hash, dev_metrics=dev, seed=c['SEED'],
                checkpoint_format_version=2, resume_supported=False,
            )
            atomic_torch_save(checkpoint, directory / 'last_cropr.pt')
            if dev[metric] > best_score:
                best_score, best_epoch = dev[metric], epoch
                atomic_torch_save(checkpoint, directory / 'best_cropr.pt')
            del checkpoint
            scheduler.step()
            print(f"  main={row['train_loss']:.4f}, aux={row['train_aux_loss']:.4f} | dev {metric}={dev[metric]:.2f} | peak={allocated:.2f} GiB")
        fit_seconds = time.perf_counter() - fit_start
        del optimizer, scheduler, initial_queries, initial_adapters, adapters, current_adapters
        checkpoint = torch.load(directory / 'best_cropr.pt', map_location='cpu', weights_only=True)
        head.load_state_dict(checkpoint['head_state_dict'])
        model.pruners.load_state_dict(checkpoint['pruner_state_dict'])
        cropr_module._load_adapter_state(model.backbone, checkpoint['lora_state_dict'])
        del checkpoint
        compute = {}
        images, target = next(iter(loaders['train']))
        images, target = images.to(self.device), target.to(self.device)
        try:
            compute.update(self.profile_training(model, images, target, task, directory))
        except Exception as exc:
            compute['training_profiling_error'] = str(exc)
            (directory / 'training_profiling_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        del images, target
        model.eval()
        merge_images = loaders['dev'].dataset[0][0].unsqueeze(0).to(self.device)
        with torch.no_grad():
            before_merge = model(merge_images, task)
            tokens_before_merge = list(model.token_stats)
            cropr_module.merge_lora(model)
            after_merge = model(merge_images, task)
            if not torch.isfinite(before_merge).all() or not torch.isfinite(after_merge).all():
                raise RuntimeError('Non-finite logits during LoRA merge validation')
            torch.testing.assert_close(after_merge, before_merge, rtol=1e-4, atol=1e-3)
            if model.token_stats != tokens_before_merge:
                raise RuntimeError('LoRA merge changed the pruning schedule')
            merge_error = float((after_merge - before_merge).abs().max())
        write_json(directory / 'lora_merge_validation.json', {
            'passed': True, 'max_absolute_logit_error': merge_error,
            'rtol': 1e-4, 'atol': 1e-3, 'images_checked': 1,
            'cropr_active': True, 'inference_lora_merged': True,
        })
        del merge_images, before_merge, after_merge
        gc.collect()
        torch.cuda.empty_cache()
        started = time.perf_counter()
        heldout = self.evaluate(model, loaders['val'], task)
        test_seconds = time.perf_counter() - started
        samples = [loaders['dev'].dataset[i] for i in range(c['BATCH_SIZE'])]
        images = torch.stack([s[0] for s in samples]).to(self.device)
        target = torch.stack([torch.as_tensor(s[1]) for s in samples]).to(self.device)
        del samples
        speed = self.benchmark(model, images, task, directory)
        try:
            compute.update(measure_flops_on_dataset(
                model, loaders['dev'].dataset, task, c['BATCH_SIZE'], c['FLOP_BATCHES'],
                c['SEED'], directory, module_names=c['ATTENTION_MODULE_NAMES'].get(name)))
        except Exception as exc:
            compute.update(whole_model_flops_coverage_ok=False, whole_model_forward_gflops_per_image=None,
                           attention_flops_coverage_ok=False, attention_forward_gflops_per_image=None,
                           flops_error=str(exc))
            (directory / 'attention_flops_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        try:
            compute.update(self.profile_model(model, images, target, task, directory))
        except Exception as exc:
            compute['profiling_error'] = str(exc)
            (directory / 'profiling_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        token_stats = list(model.token_stats)
        initial_count = token_stats[0]['input_tokens']
        token_summary = {
            'cropr_input_patch_tokens': initial_count - model.prefix,
            'cropr_min_patch_tokens': min(s['input_tokens'] for s in token_stats) - model.prefix,
            'cropr_removed_patch_tokens': sum(model.schedule.values()),
            'cropr_final_patch_tokens': token_stats[-1]['input_tokens'] - model.prefix,
            'cropr_mean_block_tokens': sum(s['input_tokens'] for s in token_stats) / len(token_stats),
            'cropr_token_processing_reduction_pct': 100 * (1 - sum(s['input_tokens'] for s in token_stats) /
                                                          (initial_count * len(token_stats))),
        }
        write_json(directory / 'tokens.json', {'blocks': token_stats, **token_summary})
        profiling_path = directory / 'profiling.json'
        profiling = json.loads(profiling_path.read_text(encoding='utf-8')) if profiling_path.exists() else {}
        write_json(profiling_path, {**profiling, **compute, **token_summary,
                                   'inference_lora_merged': True,
                                   'profile_training_scope': 'RGB forward + LoRA/head/pruner backward; no optimizer'})
        result = dict(
            model=name, task=task, status='completed', seed=c['SEED'], resolution=c['SIZE'],
            batch_size=c['BATCH_SIZE'], epochs=c['EPOCHS'][task], micro_batch_size=c['MICRO_BATCH_SIZE'],
            grad_accum_steps=c['GRAD_ACCUM_STEPS'], effective_train_batch_size=c['MICRO_BATCH_SIZE'] * c['GRAD_ACCUM_STEPS'],
            eval_batch_size=c['EVAL_BATCH_SIZE'], best_epoch=best_epoch, best_dev_metric=best_score,
            selection_metric=metric, training_mode='cropr', train_precision=c['TRAIN_PRECISION'],
            evaluation_precision='fp32', evaluation_tf32=c['EVAL_TF32'], cache_enabled=False,
            cropr_variant=c['VARIANT'], cropr_lr=c['CROPR_LR'], cropr_aux_weight=c['AUX_WEIGHT'],
            cropr_pruning_rate=c['PRUNING_RATE'], cropr_fusion=model.fusion,
            cropr_queries_per_stage=next(iter(model.pruners.values())).queries.shape[1],
            trainable_cropr_parameters=sum(p.numel() for p in model.pruners.parameters()),
            backbone_parameters=sum(p.numel() for p in model.backbone.parameters()),
            trainable_head_parameters=sum(p.numel() for p in head.parameters()),
            total_training_images=total_images, total_optimizer_steps=total_steps,
            training_seconds=sum(r['train_seconds'] for r in history),
            training_seconds_scope='RGB loading + main/auxiliary forward/backward + optimizer; excludes dev and audits',
            fit_seconds_including_dev_and_saves=fit_seconds, heldout_evaluation_seconds=test_seconds,
            train_peak_allocated_gib=max(r['train_peak_allocated_gib'] for r in history),
            train_peak_reserved_gib=max(r['train_peak_reserved_gib'] for r in history),
            inference_scope='merged LoRA + full RGB Cropr model; auxiliary aggregation/heads skipped; no feature cache',
            profile_training_scope='RGB forward + LoRA/head/pruner backward; no optimizer',
            pretrained_backbone_sha256=base_hash, frozen_state_verified=True,
            best_checkpoint_path=str(directory / 'best_cropr.pt'),
            experiment_seconds_including_measurements=time.perf_counter() - experiment_start,
            # Preserve baseline report fields with explicit N/A, never fabricated zeros.
            trainable_lora_parameters=lora_spec['trainable_parameters'], lora_rank=lora_spec['rank'],
            lora_alpha=lora_spec['alpha'], lora_lr=c['LORA_LR'], adaptation_mode='lora_cropr',
            inference_lora_merged=True, merge_max_absolute_logit_error=merge_error,
            cache_train_seconds=None, cache_dev_seconds=None, head_training_seconds=None,
            pipeline_fit_seconds_including_cache_dev_and_saves=None,
            **{f'heldout_{k}': v for k, v in heldout.items()}, **speed, **compute, **token_summary,
        )
        write_json(directory / 'result.json', result)
        return result


def run_cropr(
    *,
    ROOT=None,
    SEED=42,
    SIZE=448,
    MODEL_NAMES=('dinov2', 'siglip2', 'radio'),
    TASKS=('classification', 'segmentation'),
    VARIANT='gradual_llf',
    PRUNING_RATE=8,
    NUM_HEADS=1,
    CROPR_LR=1e-4,
    AUX_WEIGHT=1.0,
    EPOCHS=None,
    LR=None,
    WEIGHT_DECAY=0.01,
    LORA_R=8,
    LORA_ALPHA=16.0,
    LORA_LR=0.0001,
    LORA_TARGETS=None,
    BATCH_SIZE=2,
    MICRO_BATCH_SIZE=1,
    GRAD_ACCUM_STEPS=16,
    EVAL_BATCH_SIZE=1,
    TRAIN_PRECISION='bf16',
    MAX_GRAD_NORM=1.0,
    EVAL_TF32=False,
    STRICT_DETERMINISM=False,
    NUM_WORKERS=4,
    LOADER_BACKEND='thread',
    FAIL_FAST=True,
    BENCH_WARMUP=2,
    BENCH_REPEATS=5,
    BENCH_TRIALS=1,
    FLOP_BATCHES=1,
    PROFILE_REPEATS=1,
    IMAGENET_ROOT=None,
    ADE_ROOT=None,
    SPLIT_ROOT=None,
    ATTENTION_MODULE_NAMES=None,
):
    """Run real saved splits, select on dev, evaluate held-out, save baseline-compatible metrics.

    One variant per run makes task/model report rows unambiguous. All artifacts
    are under ROOT/report_2/cropr measurements. No cache, downloads, or resume.
    """
    ROOT = cropr_module.resolve_project_root(ROOT)
    EPOCHS = dict({'classification': 1, 'segmentation': 1} if EPOCHS is None else EPOCHS)
    LR = dict({'classification': 0.001, 'segmentation': 0.001} if LR is None else LR)
    IMAGENET_ROOT = Path(IMAGENET_ROOT) if IMAGENET_ROOT is not None else ROOT / 'datasets/imagenet'
    ADE_ROOT = Path(ADE_ROOT) if ADE_ROOT is not None else ROOT / 'datasets/ade20k/ADEChallengeData2016'
    SPLIT_ROOT = Path(SPLIT_ROOT) if SPLIT_ROOT is not None else ROOT / 'datasets/splits'
    ATTENTION_MODULE_NAMES = dict(ATTENTION_MODULE_NAMES or {})
    LORA_TARGETS = dict(LORA_TARGETS or {})
    if type(LORA_R) is not int or LORA_R < 1:
        raise ValueError('LORA_R must be a positive integer')
    if any(not isinstance(v, (list, tuple)) or not all(isinstance(n, str) for n in v)
           for v in LORA_TARGETS.values()):
        raise ValueError('LORA_TARGETS must map model names to exact attention projection paths')
    MODEL_NAMES, TASKS = tuple(MODEL_NAMES), tuple(TASKS)
    if not TASKS or len(set(TASKS)) != len(TASKS) or any(t not in {'classification','segmentation'} for t in TASKS):
        raise ValueError('TASKS must contain classification/segmentation without duplicates')
    if not MODEL_NAMES or len(set(MODEL_NAMES)) != len(MODEL_NAMES) or any(n not in {'dinov2','siglip2','radio'} for n in MODEL_NAMES):
        raise ValueError('Choose unique model names from dinov2, siglip2, radio')
    if VARIANT not in cropr_module.VARIANTS:
        raise ValueError('Unknown Cropr variant')
    if VARIANT == 'gradual_no_llf' and 'segmentation' in TASKS:
        raise ValueError('gradual_no_llf supports classification only; segmentation requires fusion')
    if SIZE != 448:
        raise ValueError('This run matches Report 1 at SIZE=448')
    if TRAIN_PRECISION not in {'bf16', 'fp32'} or LOADER_BACKEND not in {'thread','process'}:
        raise ValueError('Invalid precision or loader backend')
    if any(type(v) is not bool for v in (EVAL_TF32, STRICT_DETERMINISM, FAIL_FAST)):
        raise ValueError('EVAL_TF32, STRICT_DETERMINISM and FAIL_FAST must be booleans')
    if not math.isfinite(WEIGHT_DECAY) or WEIGHT_DECAY < 0:
        raise ValueError('WEIGHT_DECAY must be finite and nonnegative')
    if math.isnan(MAX_GRAD_NORM) or MAX_GRAD_NORM <= 0:
        raise ValueError('MAX_GRAD_NORM must be positive')
    for v in (CROPR_LR, AUX_WEIGHT, LORA_ALPHA, LORA_LR):
        if not math.isfinite(v) or v <= 0:
            raise ValueError('Learning rates, auxiliary weight and LoRA alpha must be positive and finite')
    for task in TASKS:
        if type(EPOCHS.get(task)) is not int or EPOCHS[task] < 1 or not math.isfinite(LR.get(task, 0)) or LR.get(task, 0) <= 0:
            raise ValueError(f'{task} requires positive epochs and learning rate')
    TRAINING_MODE = 'cropr'
    config = dict(locals())
    for key in ('BATCH_SIZE','MICRO_BATCH_SIZE','GRAD_ACCUM_STEPS','EVAL_BATCH_SIZE','NUM_HEADS',
                'BENCH_REPEATS','BENCH_TRIALS','FLOP_BATCHES','PROFILE_REPEATS'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('NUM_WORKERS','BENCH_WARMUP','PRUNING_RATE'):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    from functools import partial
    return _CroprRunner(partial(cropr_module.BaselineModel, root=ROOT), config).run()
