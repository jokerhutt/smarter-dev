"""Challenge API router for Discord bot integration.

Provides endpoints for challenge announcement management and release scheduling.
Used by the Discord bot to check for pending announcements and mark challenges as announced.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any
from uuid import UUID
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from smarter_dev.shared.database import get_db_session
from smarter_dev.web.api.dependencies import verify_api_key
from smarter_dev.web.crud import (
    CampaignOperations,
    ChallengeInputOperations,
    ChallengeSubmissionOperations,
    QuestOperations,
    SquadOperations,
    DatabaseOperationError,
    ScriptExecutionError,
)
from smarter_dev.web.models import Campaign

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quests", tags=["quests"])


@router.get("/{quest_id}")
async def get_quest(
    quest_id: UUID,
    session: AsyncSession = Depends(get_db_session),
    api_key=Depends(verify_api_key),
) -> Dict[str, Dict[str, Any]]:
    """Get a challenge with its campaign data.

    Args:
        quest: UUID of the quest to retrieve

    Returns:
        Quest data with quest information
    """
    try:
        campaign_ops = QuestOperations(session)
        quest = await campaign_ops.get_challenge_with_campaign(challenge_id)

        if not quest:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Challenge not found"
            )

        campaign = quest.campaign
        quest_data = {
            "id": str(quest.id),
            "title": quest.title,
            "description": quest.description,
            "order_position": quest.order_position,
            "is_released": quest.is_released,
            "is_announced": quest.is_announced,
            "released_at": quest.released_at.isoformat() if quest.released_at else None,
            "announced_at": (
                quest.announced_at.isoformat() if quest.announced_at else None
            ),
            "created_at": quest.created_at.isoformat(),
            "guild_id": campaign.guild_id,
            "announcement_channels": campaign.announcement_channels,
            "campaign": {
                "id": str(campaign.id),
                "title": campaign.title,
                "start_time": campaign.start_time.isoformat(),
                "release_cadence_hours": campaign.release_cadence_hours,
                "is_active": campaign.is_active,
            },
        }

        return {"challenge": quest_data}

    except DatabaseOperationError as e:
        logger.error(f"Database error getting challenge: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve challenge",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting challenge: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
