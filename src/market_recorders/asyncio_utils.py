from __future__ import annotations

import asyncio
from typing import Any


async def cancel_and_await(*tasks: asyncio.Task[Any] | None) -> bool:
    """Cancel child tasks and finish their cleanup before re-raising cancellation."""

    pending = [task for task in tasks if task is not None]
    for task in pending:
        task.cancel()
    if not pending:
        return False

    waiter = asyncio.gather(*pending, return_exceptions=True)
    cancelled_during_cleanup = False
    while True:
        try:
            await asyncio.shield(waiter)
            break
        except asyncio.CancelledError:
            cancelled_during_cleanup = True
            if waiter.done():
                break
    return cancelled_during_cleanup
