"""Upstream router — resolves API key to target URL via YAML config."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import yaml

log = logging.getLogger(__name__)


@dataclass
class ResolvedTarget:
    target_url: str
    upstream_name: str
    matched_key_prefix: str | None  # None = default fallback
    path_prefix: str | None = None  # e.g. "/v1/messages" to override upstream path for Messages API


class UpstreamRouter:
    """Routes API keys to upstream targets by prefix matching.

    Matching strategy: iterates upstreams in config order, and within each
    upstream iterates keys in list order. The **first** matching prefix wins
    (NOT longest-prefix-first). Place more specific prefixes before generic
    ones within the same upstream to get correct routing.
    """

    def __init__(self, config_path: str | None) -> None:
        self._config_path = config_path
        self._default: str = ""
        self._upstreams: list[tuple[str, str, list[str], str | None]] = []  # [(name, target, [prefixes], path_prefix)]
        if config_path:
            self._load(config_path)

    @property
    def is_available(self) -> bool:
        return bool(self._upstreams)

    def _load(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            log.warning(f"[upstream] config file not found: {path}, router disabled")
            return
        except Exception as exc:
            log.error(f"[upstream] failed to load config {path}: {exc}")
            return

        self._default = data.get("default", "")
        upstreams_data = data.get("upstreams", {})
        self._upstreams = []
        for name, cfg in upstreams_data.items():
            target = cfg.get("target", "")
            keys = cfg.get("keys", [])
            if target and keys:
                self._upstreams.append((name, target, keys, cfg.get("path_prefix")))

        names = [n for n, _, _, _ in self._upstreams]
        log.info(f"[upstream] loaded {len(self._upstreams)} upstreams: {names}, default={self._default}")

    def reload(self) -> tuple[bool, str]:
        """运行时重载配置文件（原子替换，失败保持旧配置）。

        Returns:
            (ok, message): ok=False 时保留旧配置继续服务（不破坏运行中请求）。
        """
        if not self._config_path:
            return False, "no config path (router disabled)"
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            new_default = data.get("default", "")
            new_upstreams: list[tuple[str, str, list[str], str | None]] = []
            for name, cfg in (data.get("upstreams", {}) or {}).items():
                target = cfg.get("target", "")
                keys = cfg.get("keys", [])
                if target and keys:
                    new_upstreams.append((name, target, keys, cfg.get("path_prefix")))
            if not new_upstreams and self._upstreams:
                return False, "new config has no upstreams (refusing to empty router)"
            # 原子替换：resolve 每次读 _upstreams/_default（GIL 下引用替换对读线程安全）
            self._default = new_default
            self._upstreams = new_upstreams
            names = [n for n, _, _, _ in self._upstreams]
            log.info(f"[upstream] reloaded {len(self._upstreams)} upstreams: {names}, default={self._default}")
            return True, f"reloaded {len(new_upstreams)} upstreams: {names}, default={new_default}"
        except FileNotFoundError:
            return False, f"config file not found: {self._config_path}"
        except Exception as exc:
            return False, f"failed to reload config: {exc}"

    def resolve(self, auth_header: str | None) -> ResolvedTarget | None:
        if not self._upstreams:
            return None

        key = self._extract_key(auth_header)

        for name, target, prefixes, path_prefix in self._upstreams:
            for prefix in prefixes:
                if key and key.startswith(prefix):
                    log.debug(f"[upstream] key={key[:12]}** → {name} (prefix \"{prefix}\")")
                    return ResolvedTarget(
                        target_url=target,
                        upstream_name=name,
                        matched_key_prefix=prefix,
                        path_prefix=path_prefix,
                    )

        log.debug(f"[upstream] key={key[:12] if key else '<none>'}** → {self._default} (default)")
        return ResolvedTarget(
            target_url=self._resolve_default_target(),
            upstream_name=self._default,
            matched_key_prefix=None,
        )

    def _extract_key(self, auth_header: str | None) -> str | None:
        if not auth_header:
            return None
        prefix = "Bearer "
        if auth_header.startswith(prefix):
            return auth_header[len(prefix):]
        return auth_header

    def _resolve_default_target(self) -> str:
        for name, target, _, _ in self._upstreams:
            if name == self._default:
                return target
        return self._upstreams[0][1] if self._upstreams else ""
