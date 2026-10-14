"""Reversible runtime integration for HARP and RHT quantized projections."""

from collections import Counter
import copy
import types
import weakref
import torch
from harp_runtime.plan import RotationPlan, extension
from .codebooks import CODEBOOK_BITS, decode_three

_MISSING = object()


def _version(t):
    try:
        return t._version
    except RuntimeError:

        return None


class _RotationState:
    def __init__(self, rot, name, prefill):
        self.rot = weakref.ref(rot)
        self.name, self.prefill = name, prefill
        self.plan = None
        self.signature = None
        self.guard = None
        self.calls = Counter()
        self.rebuilds = 0

    def invalidate(self):
        self.plan = None
        self.signature = None
        self.guard = None
        rot = self.rot()
        if rot is not None:
            rot._blocks_cache.clear()

    def fingerprint(self):
        rot = self.rot()

        tensors = [rot.theta_is_quantized]
        tensors.extend(t for p in rot.theta_raw for t in p)
        for t in range(rot.m):
            tensors.append(getattr(rot, f"M_fixed_{t}"))
            if hasattr(rot, f"triu_idx_{t}"):
                tensors.append(getattr(rot, f"triu_idx_{t}"))
        for p in range(rot.n_passes):
            for t in range(rot.m):
                tensors.extend(
                    [getattr(rot, rot._qname(p, t)), getattr(rot, rot._sname(p, t))]
                )
        versions = tuple((id(t), _version(t)) for t in tensors)
        if any(v is None for _, v in versions):
            return None
        return (
            rot.n,
            tuple(rot.stage_bs),
            rot.n_passes,
            rot.ordering_mode,
            rot.theta_clip,
            tuple(rot.stage_is_givens),
            versions,
        )

    def get_plan(self):
        rot = self.rot()
        if self.plan is not None and self.guard is not None and self.guard.unchanged():
            return self.plan
        signature = self.fingerprint()
        if signature is None:

            self.calls["fallback_unversioned_parameters"] += 1
            self.invalidate()
            return None
        if self.plan is not None and self.guard is None and signature == self.signature:
            return self.plan
        self.invalidate()
        if rot.ordering_mode != "stride":
            self.calls["fallback_ordering"] += 1
            return None
        device = next(rot.parameters()).device

        with torch.inference_mode(False), torch.no_grad():
            passes = [
                [
                    rot._get_blocks(p, t, device, torch.float32).detach()
                    for t in range(rot.m)
                ]
                for p in range(rot.n_passes)
            ]
            self.plan = RotationPlan(passes, prefill=self.prefill)
        if self.plan.fused_eligible:
            bindings = []
            for module in rot.modules():
                for attr in ("_parameters", "_buffers", "_modules"):
                    mapping = getattr(module, attr)
                    bindings.append(
                        (
                            weakref.ref(module),
                            attr,
                            {
                                key: weakref.ref(value) if value is not None else None
                                for key, value in mapping.items()
                            },
                        )
                    )

            values = [
                (weakref.ref(rot), key, copy.deepcopy(getattr(rot, key)))
                for key in (
                    "n",
                    "stage_bs",
                    "n_passes",
                    "ordering_mode",
                    "theta_clip",
                    "stage_is_givens",
                )
            ]
            tensors = list(rot.parameters()) + list(rot.buffers())
            self.guard = extension().DependencyGuard(tensors, bindings, values)
        self.signature = signature
        self.rebuilds += 1
        return self.plan

    def boundary(self, x, scale, inverse, mode, multiplier):
        plan = self.plan
        if plan is None:
            plan = self.get_plan()
        if plan is None or not plan.fused_eligible:
            return None
        if plan._cpp is None:
            plan.prepare()
        y = plan._cpp.apply_guarded(x, scale, inverse, mode, multiplier, self.guard)
        if y is None:
            plan = self.get_plan()
            if plan is None or not plan.fused_eligible:
                return None
            plan.prepare()
            y = plan._cpp.apply(x, scale, inverse, mode, multiplier)
        plan.last_path = "cuda_boundary"
        return y

    def execute(self, x, inverse, original):
        rot = self.rot()
        if (
            torch.is_grad_enabled()
            or rot.training
            or torch.is_autocast_enabled()
            or not x.is_cuda
            or x.dtype != torch.float32
            or x.numel() == 0
            or (not self.prefill and x.numel() // rot.n > 8)
        ):
            self.calls["fallback_reference_context"] += 1

            sig = self.fingerprint()
            if sig is None or sig != self.signature:
                self.invalidate()
            return original(x)
        plan = self.get_plan()
        if plan is None:
            return original(x)
        y = plan.apply(x, inverse)
        self.calls[plan.last_path] += 1
        return y


class AccelerationHandle:
    """Owns reversible adapters, derived plans, and execution coverage."""

    def __init__(self, model, *, prefill="auto", boundaries=True, shared_core=False):
        from lib.incoherence.harp_lib import FactorizedOrthogonalRotation
        from lib.linear.quantized_linear import QuantizedLinear

        self.model = weakref.ref(model)
        self.prefill, self.boundaries = prefill, boundaries
        if prefill not in (True, False, "auto"):
            raise ValueError('prefill must be True, False or "auto"')
        self.shared_core = shared_core
        self.native_core_only = any(
            p.dtype == torch.bfloat16 for p in model.parameters()
        )
        self.resolved_prefill = self._prefill_enabled(model)
        self.active = True
        self.states, self.projections, self._patches = {}, {}, []
        for name, rot in model.named_modules():
            if not isinstance(rot, FactorizedOrthogonalRotation):
                continue
            state = _RotationState(rot, name, self.resolved_prefill)
            self.states[id(rot)] = state
            forward, inverse = rot.transform, rot.inverse_transform

            def transform(module, x, _s=state, _f=forward):
                if module is not _s.rot():
                    raise RuntimeError(
                        "Remove HARP acceleration before deepcopy; reinstall on the copy"
                    )
                return _s.execute(x, False, _f)

            def inverse_transform(module, x, _s=state, _f=inverse):
                if module is not _s.rot():
                    raise RuntimeError(
                        "Remove HARP acceleration before deepcopy; reinstall on the copy"
                    )
                return _s.execute(x, True, _f)

            self._patch(rot, "transform", transform)
            self._patch(rot, "inverse_transform", inverse_transform)

            for method in (
                "_apply",
                "_load_from_state_dict",
                "quantize_theta_8bit",
                "train",
            ):
                old = getattr(rot, method)

                def change(module, *args, _s=state, _old=old, **kwargs):
                    _s.invalidate()
                    try:
                        return _old(*args, **kwargs)
                    finally:
                        _s.invalidate()

                self._patch(rot, method, change)
        if boundaries:
            for name, layer in model.named_modules():
                if not isinstance(layer, QuantizedLinear) or layer.incoh_mode not in (
                    "harp",
                    "had",
                ):
                    continue
                stats = dict(
                    name=name,
                    layer=weakref.ref(layer),
                    calls=Counter(),
                    reason="not_initialized",
                )
                self.projections[id(layer)] = stats
                old = layer.no_ckpt_forward
                stats["no_ckpt_original"] = old

                def linear(module, x, _old=old, _stats=stats):
                    if module is not _old.__self__:
                        raise RuntimeError(
                            "Remove HARP acceleration before deepcopy; reinstall on the copy"
                        )
                    if module.built_codebook_class:
                        self._attach_codebook(module, _stats)
                    y = _old(x)
                    if module.built_codebook_class:
                        self._attach_codebook(module, _stats)
                    return y

                stats["lazy_function"] = linear
                self._patch(layer, "no_ckpt_forward", linear)
                if layer.built_codebook_class:
                    self._attach_codebook(layer, stats)

        for method in ("_apply", "load_state_dict"):
            old = getattr(model, method)

            def root_change(module, *args, _old=old, **kwargs):
                try:
                    return _old(*args, **kwargs)
                finally:
                    self.refresh()

            self._patch(model, method, root_change)

    def _prefill_enabled(self, model):

        return (
            not self.native_core_only if self.prefill == "auto" else bool(self.prefill)
        )

    def _patch(self, obj, name, function):
        previous = obj.__dict__.get(name, _MISSING)
        setattr(obj, name, types.MethodType(function, obj))
        self._patches.append((obj, name, previous))

    def _attach_codebook(self, layer, stats):
        cb = layer.codebook_class

        layer.no_ckpt_forward = stats["no_ckpt_original"]
        if getattr(cb, "_harp_runtime_owner", None) is self:
            return
        module = cb.__class__.__module__
        if module not in CODEBOOK_BITS:
            stats["reason"] = "codebook_without_boundary_adapter"
            return
        if layer.incoh_mode == "had":
            from .rht import attach

            attach(self, layer, stats, CODEBOOK_BITS[module])
            return
        pre_rot = getattr(layer.harp.harp_v, "rot", layer.harp.harp_v)
        post_rot = getattr(layer.harp.harp_u, "rot", layer.harp.harp_u)
        pre_state, post_state = self.states.get(id(pre_rot)), self.states.get(
            id(post_rot)
        )
        if pre_state is None or post_state is None:
            stats["reason"] = "composite_rotation_uses_inner_acceleration"
            return
        four = CODEBOOK_BITS[module] == 4
        three = CODEBOOK_BITS[module] == 3
        original = cb.forward
        decode_cached = None
        stats["rotation_ids"] = (id(pre_rot), id(post_rot))

        def forward(codebook, x, indices, su, sv, *args, **kwargs):
            nonlocal decode_cached
            if codebook is not original.__self__:
                raise RuntimeError(
                    "Remove HARP acceleration before deepcopy; reinstall on the copy"
                )
            rows = x.numel() // len(su)
            current_harp = layer._modules.get("harp")
            if layer.incoh_mode != "harp" or current_harp is None:
                stats["calls"]["fallback_native_mode"] += 1
                return original(x, indices, su, sv, *args, **kwargs)
            current_v = current_harp._modules.get("harp_v")
            current_u = current_harp._modules.get("harp_u")
            current_pre = (
                current_v._modules.get("rot", current_v)
                if current_v is not None
                else None
            )
            current_post = (
                current_u._modules.get("rot", current_u)
                if current_u is not None
                else None
            )
            if current_pre is not pre_rot or current_post is not post_rot:
                stats["calls"]["fallback_replaced_rotation"] += 1
                return original(x, indices, su, sv, *args, **kwargs)
            if (
                len(args) != 4
                or torch.is_grad_enabled()
                or layer.training
                or torch.is_autocast_enabled()
                or not x.is_cuda
                or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or not 0 < rows <= 8
                or kwargs.get("train_mode", False)
                or kwargs.get("rescale_WH", False)
                or kwargs.get("rank", 0) > 0
                or kwargs.get("harp") is not current_harp
                or kwargs.get("incoh_mode") != "harp"
                or su.dtype != torch.float16
                or sv.dtype != torch.float16
            ):
                stats["calls"]["explicit_boundaries"] += 1
                return original(x, indices, su, sv, *args, **kwargs)
            n, m = len(su), len(sv)
            inp = x.float() if x.dtype == torch.bfloat16 else x
            v = pre_state.boundary(
                inp.reshape(-1, n).contiguous(),
                su.contiguous(),
                False,
                1,
                codebook.scale,
            )
            if v is None:
                stats["calls"]["explicit_boundaries"] += 1
                return original(x, indices, su, sv, *args, **kwargs)
            idx0 = indices[0].view(m // 16, n // 64, 8, 4)
            grid = codebook.codebook.grid_packed_abs
            shared_eligible = (
                not self.native_core_only
                and x.dtype != torch.bfloat16
                and pre_state.plan.cuda_major >= 8
                and n < 65536
                and m < 65536
            )
            if three:
                use_shared = self.shared_core and rows > 1 and shared_eligible
                if use_shared and decode_cached is None:
                    from .shared import decoder

                    decode_cached = decoder()
                override = kwargs.get("resid_scale_override", -1)
                residual_scale = (
                    override if override > 0 else codebook.codebook.opt_resid_scale
                )
                y = decode_three(
                    v,
                    indices,
                    codebook.codebook,
                    residual_scale,
                    decode_cached if use_shared else None,
                )
                if use_shared:
                    stats["calls"]["shared_packed_core"] += 1
            elif self.shared_core and rows > 1 and shared_eligible:
                if decode_cached is None:
                    from .shared import decoder

                    decode_cached = decoder()
                decode = decode_cached
                y = decode(v, idx0, grid)
                if four:
                    override = kwargs.get("resid_scale_override", -1)
                    rs = override if override > 0 else codebook.codebook.opt_resid_scale
                    y = y + decode(
                        v / rs, indices[1].view(m // 16, n // 64, 8, 4), grid
                    )
                y = y.half()
                stats["calls"]["shared_packed_core"] += 1
            elif four:
                import quiptools_cuda

                override = kwargs.get("resid_scale_override", -1)
                rs = override if override > 0 else codebook.codebook.opt_resid_scale
                idx1 = indices[1].view(m // 16, n // 64, 8, 4)
                if rows == 1:
                    y = (
                        torch.ops.quip_lib.decode_matvec_e8p(v[0], idx0, grid, m, n)
                        + torch.ops.quip_lib.decode_matvec_e8p(
                            v[0] / rs, idx1, grid, m, n
                        )
                    ).unsqueeze(0)
                else:
                    w0 = quiptools_cuda.decompress_packed_e8p(idx0, grid)
                    w1 = quiptools_cuda.decompress_packed_e8p(idx1, grid)
                    y = v @ (w0 + w1 / rs).T
            elif rows == 1:
                y = torch.ops.quip_lib.decode_matvec_e8p(
                    v[0], idx0, grid, m, n
                ).unsqueeze(0)
            else:
                w = torch.ops.quip_lib.decompress_packed_e8p(idx0, grid, m, n)
                y = v @ w.T
            z = post_state.boundary(
                y.contiguous(),
                sv.contiguous(),
                True,
                2 if x.dtype == torch.float16 else 3,
                codebook.scale,
            )
            if z is None:
                stats["calls"]["explicit_boundaries"] += 1
                return original(x, indices, su, sv, *args, **kwargs)
            if self.shared_core and rows > 1 and not shared_eligible:
                stats["calls"]["native_core_precision_or_geometry_policy"] += 1
            stats["calls"]["cuda_boundary"] += 1
            return z.reshape(*x.shape[:-1], m)

        self._patch(cb, "forward", forward)
        object.__setattr__(cb, "_harp_runtime_owner", self)
        self._patches.append((cb, "_harp_runtime_owner", _MISSING))
        stats["reason"] = "adapter_installed"

    def refresh(self):
        """Discard derived plans after external edits (e.g. unsafe .data writes)."""
        model = self.model()
        if model is not None:
            self.native_core_only = any(
                p.dtype == torch.bfloat16 for p in model.parameters()
            )
            self.resolved_prefill = self._prefill_enabled(model)

        for stats in self.projections.values():
            layer = stats["layer"]()
            if layer is not None and layer.built_codebook_class and self.active:
                if layer.Qidxs_list is not None:
                    layer.Qidxs_list = [
                        getattr(layer, f"Qidxs_{i}")
                        for i in range(len(layer.Qidxs_list))
                    ]
                self._attach_codebook(layer, stats)
            elif layer is not None and self.active:
                layer.no_ckpt_forward = types.MethodType(stats["lazy_function"], layer)
        for state in self.states.values():
            state.prefill = self.resolved_prefill
            state.invalidate()
        return self

    def report(self):

        rotation_calls = {
            key: Counter(state.calls) for key, state in self.states.items()
        }
        for projection in self.projections.values():
            hits = projection["calls"]["cuda_boundary"]
            for key in projection.get("rotation_ids", ()):
                rotation_calls[key]["cuda_boundary"] += hits
        return dict(
            active=self.active,
            shared_core_allowed=not self.native_core_only,
            prefill_requested=self.prefill,
            prefill_enabled=self.resolved_prefill,
            rotations=[
                dict(
                    name=s.name,
                    calls=dict(rotation_calls[key]),
                    rebuilds=s.rebuilds,
                    plan=s.plan.description() if s.plan is not None else None,
                )
                for key, s in self.states.items()
            ],
            projections=[
                dict(name=p["name"], reason=p["reason"], calls=dict(p["calls"]))
                for p in self.projections.values()
            ],
            counters_scope="Python executions; CUDA graph replays do not increment counters",
        )

    def reset_statistics(self):
        for state in self.states.values():
            state.calls.clear()
        for p in self.projections.values():
            p["calls"].clear()

    def remove(self):
        if not self.active:
            return
        for obj, name, previous in reversed(self._patches):
            if previous is _MISSING:
                obj.__dict__.pop(name, None)
            else:
                setattr(obj, name, previous)
        self._patches.clear()

        for stats in self.projections.values():
            stats.pop("no_ckpt_original", None)
            stats.pop("lazy_function", None)
        self.active = False
        self.refresh()
        model = self.model()
        if model is not None:
            model.__dict__.pop("_harp_runtime_handle", None)
        self.active = False


def accelerate(model, *, prefill="auto", boundaries=True, shared_core=False):
    """Accelerate discovered HARP and RHT operators during inference."""
    existing = getattr(model, "_harp_runtime_handle", None)
    if existing is not None and existing.active:
        if existing.model() is not model:
            raise RuntimeError(
                "Remove HARP acceleration before deepcopy; reinstall on the copy"
            )
        if (existing.prefill, existing.boundaries, existing.shared_core) != (
            prefill,
            boundaries,
            shared_core,
        ):
            raise ValueError("Remove the existing handle before changing options")
        return existing
    handle = AccelerationHandle(
        model, prefill=prefill, boundaries=boundaries, shared_core=shared_core
    )
    object.__setattr__(model, "_harp_runtime_handle", handle)
    return handle
