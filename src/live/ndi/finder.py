"""NDI 6 Source Discovery: Background network scanner for active NDI streams."""

from __future__ import annotations

import ctypes
import threading
import time
from typing import Callable

from .runtime import (
    NDIlib_find_create_t,
    NDIlib_source_t,
    get_ndi_lib,
)


class NdiSourceFinder:
    """Asynchronous background finder detecting NDI broadcast streams on the local network."""

    def __init__(self, on_sources_changed: Callable[[list[tuple[str, str]]], None] | None = None) -> None:
        self._on_changed = on_sources_changed
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._instance: int | None = None
        self._sources: list[tuple[str, str]] = []

    def start(self) -> None:
        """Start the background discovery scanner."""
        with self._lock:
            if self._running:
                return
            self._running = True

            lib = get_ndi_lib()
            settings = NDIlib_find_create_t(
                show_local_sources=True,
                p_groups=None,
                p_extra_ips=None,
            )
            instance = lib.NDIlib_find_create_v2(ctypes.byref(settings))
            if not instance:
                self._running = False
                raise RuntimeError("Failed to create NDI finder instance.")
            self._instance = instance

            self._thread = threading.Thread(
                target=self._scan_loop,
                name="dlss5-ndi-finder",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop scanner and release finder resources cleanly."""
        with self._lock:
            self._running = False
            instance = self._instance
            self._instance = None

        if instance:
            lib = get_ndi_lib()
            lib.NDIlib_find_destroy(instance)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_sources(self) -> list[tuple[str, str]]:
        """Return the current list of discovered sources as (name, url_address)."""
        with self._lock:
            return list(self._sources)

    def _scan_loop(self) -> None:
        lib = get_ndi_lib()
        while self._running and self._instance:
            # Wait up to 1000ms for network source state changes
            lib.NDIlib_find_wait_for_sources(self._instance, 1000)
            if not self._running or not self._instance:
                break

            count = ctypes.c_uint32(0)
            sources_ptr = lib.NDIlib_find_get_current_sources(self._instance, ctypes.byref(count))

            new_list: list[tuple[str, str]] = []
            if sources_ptr and count.value > 0:
                for i in range(count.value):
                    src = sources_ptr[i]
                    name = src.name()
                    addr = src.address()
                    if name:
                        new_list.append((name, addr))

            changed = False
            with self._lock:
                if new_list != self._sources:
                    self._sources = new_list
                    changed = True

            if changed and self._on_changed:
                try:
                    self._on_changed(list(new_list))
                except Exception:
                    pass
