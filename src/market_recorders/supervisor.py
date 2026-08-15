from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


class FatalServiceError(RuntimeError):
    """A service failed in a way that should stop the whole supervisor."""


@dataclass(slots=True)
class ManagedServiceState:
    name: str
    summary: dict[str, Any]
    factory: Callable[[], Any]
    status: str = "pending"
    run_count: int = 0
    restart_count: int = 0
    consecutive_failures: int = 0
    last_error: str | None = None
    runner_task: asyncio.Task[None] | None = None
    stop_requested: bool = False
    fatal: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "run_count": self.run_count,
            "restart_count": self.restart_count,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            **self.summary,
        }


@dataclass(slots=True)
class RecorderSupervisor:
    logger: logging.Logger
    restart_min_seconds: float = 1.0
    restart_max_seconds: float = 30.0
    restart_jitter_seconds: float = 0.5
    disabled_components: list[str] = field(default_factory=list)
    _services: list[ManagedServiceState] = field(default_factory=list)
    _stop_reason: str = "completed"
    _fatal_error: str | None = None
    _loop: asyncio.AbstractEventLoop | None = None
    _shutdown_event: asyncio.Event | None = None

    def add_service(self, name: str, summary: dict[str, Any], factory: Callable[[], Any]) -> None:
        self._services.append(ManagedServiceState(name=name, summary=dict(summary), factory=factory))

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._shutdown_event = asyncio.Event()
        self.logger.info(
            "Starting recorder supervisor",
            extra={
                "service_count": len(self._services),
                "services": [state.snapshot() for state in self._services],
                "disabled_components": self.disabled_components,
                "restart_policy": {
                    "min_seconds": self.restart_min_seconds,
                    "max_seconds": self.restart_max_seconds,
                    "jitter_seconds": self.restart_jitter_seconds,
                },
            },
        )
        for state in self._services:
            state.runner_task = asyncio.create_task(self._run_service_loop(state), name=f"svc:{state.name}")
        try:
            if self._services:
                service_group = asyncio.gather(*(state.runner_task for state in self._services if state.runner_task))
                shutdown_wait = asyncio.create_task(self._shutdown_event.wait(), name="supervisor:shutdown_wait")
                done, pending = await asyncio.wait(
                    {service_group, shutdown_wait},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if shutdown_wait in done and not service_group.done():
                    await self._stop_all()
                    await asyncio.gather(service_group, return_exceptions=True)
                else:
                    shutdown_wait.cancel()
                    await asyncio.gather(shutdown_wait, return_exceptions=True)
                    await service_group
            else:
                self.logger.warning("Recorder supervisor started with zero enabled services")
        except asyncio.CancelledError:
            self._stop_reason = "cancelled"
            raise
        finally:
            await self._stop_all()
            self.logger.info(
                "Recorder supervisor stopped",
                extra={
                    "stop_reason": self._stop_reason,
                    "fatal_error": self._fatal_error,
                    "services": [state.snapshot() for state in self._services],
                },
            )
            self._shutdown_event = None
            self._loop = None

    def request_stop(self, *, reason: str = "requested_stop") -> None:
        if self._stop_reason == "completed":
            self._stop_reason = reason
        for state in self._services:
            state.stop_requested = True
        if self._loop is not None and self._shutdown_event is not None:
            self._loop.call_soon_threadsafe(self._shutdown_event.set)

    async def _run_service_loop(self, state: ManagedServiceState) -> None:
        while not state.stop_requested:
            state.run_count += 1
            state.status = "running"
            self.logger.info(
                "Service run starting",
                extra={"service_name": state.name, "run_count": state.run_count, **state.summary},
            )
            try:
                await self._invoke_factory(state.factory)
            except FatalServiceError as exc:
                state.status = "fatal"
                state.last_error = repr(exc)
                state.fatal = True
                self._stop_reason = "fatal_service_error"
                self._fatal_error = f"{state.name}: {exc!r}"
                self.logger.exception(
                    "Service failed fatally",
                    extra={"service_name": state.name, "fatal": True, "error": repr(exc), **state.summary},
                )
                await self._request_stop_all(excluding=state.name)
                return
            except asyncio.CancelledError:
                state.status = "cancelled"
                raise
            except Exception as exc:
                state.status = "crashed"
                state.consecutive_failures += 1
                state.restart_count += 1
                state.last_error = repr(exc)
                self.logger.exception(
                    "Service crashed and will be restarted",
                    extra={
                        "service_name": state.name,
                        "restart_count": state.restart_count,
                        "consecutive_failures": state.consecutive_failures,
                        "error": repr(exc),
                        **state.summary,
                    },
                )
                delay = min(
                    self.restart_max_seconds,
                    self.restart_min_seconds * max(1, state.consecutive_failures),
                )
                delay += random.random() * self.restart_jitter_seconds
                await asyncio.sleep(delay)
                continue
            else:
                state.status = "completed"
                state.consecutive_failures = 0
                state.last_error = None
                return

    async def _invoke_factory(self, factory: Callable[[], Any]) -> None:
        created = factory()
        if hasattr(created, "run"):
            result = created.run()
        else:
            result = created
        if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
            await result
            return
        if callable(result):
            maybe = result()
            if asyncio.iscoroutine(maybe) or isinstance(maybe, Awaitable):
                await maybe

    async def _request_stop_all(self, *, excluding: str | None = None) -> None:
        for state in self._services:
            if excluding is not None and state.name == excluding:
                continue
            state.stop_requested = True
            if state.runner_task is not None:
                state.runner_task.cancel()

    async def _stop_all(self) -> None:
        for state in self._services:
            state.stop_requested = True
        tasks = [state.runner_task for state in self._services if state.runner_task and not state.runner_task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
