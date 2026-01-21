# rag/models/log_config.py

import logging
import json
import sys
import os
from contextvars import ContextVar

# Context variables for dynamic logging context
request_id_ctx = ContextVar("request_id", default=None)
agent_name_ctx = ContextVar("agent_name", default=None)
pipeline_stage_ctx = ContextVar("pipeline_stage", default=None)

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "request_id": request_id_ctx.get(),
            "agent": agent_name_ctx.get(),
            "stage": pipeline_stage_ctx.get(),
        }
        return json.dumps(log_record)

def setup_json_logging(log_file_path="logs/pipeline.log"):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    handler.setFormatter(JsonFormatter())

    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
        force=True  # ensures reconfiguration if logging was already set
    )
