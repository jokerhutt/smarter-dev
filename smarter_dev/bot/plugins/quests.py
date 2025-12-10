from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status, Query
import logging
from typing import TYPE_CHECKING
import hikari
import lightbulb

router = APIRouter(prefix="/quests", tags=["quests"])

plugin = lightbulb.Plugin("quests")


@plugin.command
@lightbulb.command("quests", "Quest related commands")
@lightbulb.implements(lightbulb.SlashCommandGroup)
async def quests_group() -> None:
    """Quests command group."""
    pass


def load(bot: lightbulb.BotApp) -> None:
    """Load the challenges plugin."""
    bot.add_plugin(plugin)


def unload(bot: lightbulb.BotApp) -> None:
    """Unload the challenges plugin."""
    bot.remove_plugin(plugin)
