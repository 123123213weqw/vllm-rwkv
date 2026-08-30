# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError):
        return False


def compatibility() -> dict[str, object]:
    required_modules = (
        "vllm.model_executor.models.rwkv7",
        "vllm.v1.core.sched.rwkv_decode_wave",
        "vllm.v1.worker.gpu.model_states.rwkv",
        "vllm.rwkv_stateful.snapshot",
    )
    modules = {name: _module_present(name) for name in required_modules}
    return {
        "compatible": all(modules.values()),
        "distribution": "rwkv-vllm-native",
        "modules": modules,
    }


def require_compatible() -> None:
    report = compatibility()
    if not report["compatible"]:
        missing = [name for name, present in report["modules"].items() if not present]
        raise RuntimeError(
            "vllm-rwkv7-stateful requires the rwkv-vllm-native scheduler and "
            f"model-state hooks; missing {missing}."
        )


def register() -> None:
    require_compatible()
    from vllm import ModelRegistry

    if "RWKV7ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "RWKV7ForCausalLM",
            "vllm.model_executor.models.rwkv7:RWKV7ForCausalLM",
        )


def doctor_main() -> None:
    report = compatibility()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["compatible"]:
        raise SystemExit(1)


__all__ = ["compatibility", "doctor_main", "register", "require_compatible"]
