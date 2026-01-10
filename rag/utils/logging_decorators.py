
# utils/logging_decorators.py

import functools
import uuid
import logging
from ..models  import request_id_ctx, agent_name_ctx, pipeline_stage_ctx

def with_logging_context(stage: str, agent: str = None):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            request_id_ctx.set(str(uuid.uuid4()))
            pipeline_stage_ctx.set(stage)
            if agent:
                agent_name_ctx.set(agent)
            logging.info(f"Entering stage: {stage} | Agent: {agent}")
            try:
                return await func(*args, **kwargs)
            finally:
                logging.info(f"Exiting stage: {stage} | Agent: {agent}")
                # Optional: clear context after execution
                request_id_ctx.set(None)
                pipeline_stage_ctx.set(None)
                agent_name_ctx.set(None)
        return wrapper
    return decorator
