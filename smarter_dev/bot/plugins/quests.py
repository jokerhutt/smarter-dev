from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, status, Query
import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from smarter_dev.bot.services.api_client import APIClient
import hikari
import lightbulb
import logging
from smarter_dev.shared.config import Settings, get_settings

router = APIRouter(prefix="/quests", tags=["quests"])

plugin = lightbulb.Plugin("quests")

logger = logging.getLogger(__name__)
settings = get_settings()

## Abstractions
async def defer_ephemeral(ctx):
    await ctx.respond(
        hikari.ResponseType.DEFERRED_MESSAGE_CREATE,
        flags=hikari.MessageFlag.EPHEMERAL,
    )

async def get_guild_id()


@plugin.command
@lightbulb.command("quests", "Quest related commands")
@lightbulb.implements(lightbulb.SlashCommandGroup)
async def quests_group() -> None:
    """Quests command group."""
    pass

def initialize_client (settings: Settings, default_timeout = 30) :
    return APIClient(base_url=settings.api_base_url, api_key=settings.bot_api_key, default_timeout=default_timeout)


@quests_group.child
@lightbulb.command("event", "View current quest event/campaign information")
@lightbulb.implements(lightbulb.SlashSubCommand)
async def event_command(ctx: lightbulb.Context) -> None:
    """Display current quest information."""
    try:
        await defer_ephemeral(ctx)

        # Get guild ID
        if ctx.guild_id is None:
            await ctx.edit_last_response("This command can only be used in a server.")
            return

        # Initialize API client
        initialize_client(settings)

    except Exception as fatal_error:
        logger.error(f"Fatal error in event command: {fatal_error}")
        try:
            await ctx.edit_last_response("A fatal error occurred. Please try again later.")
        except Exception:
            pass


def load(bot: lightbulb.BotApp) -> None:
    """Load the challenges plugin."""
    bot.add_plugin(plugin)


def unload(bot: lightbulb.BotApp) -> None:
    """Unload the challenges plugin."""
    bot.remove_plugin(plugin)
