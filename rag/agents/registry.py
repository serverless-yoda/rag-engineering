# agents/registry.py

"""
🧭 AgentRegistry: Central registry for agent lookup and capability description.

This module maps agent names to their handler classes and provides a structured
description of agent capabilities for use by the PlannerAgent.
"""
import logging
from inspect import signature

class AgentRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str, capabilities: str = "", required_inputs: list = None):
        """Decorator to register an agent class dynamically."""
        def decorator(agent_class):
            logging.info(f"✅ Registered agent: {name}")
            cls._registry[name.lower()] = {
                "class": agent_class,
                "capabilities": capabilities,
                "required_inputs": required_inputs or []
            }
            return agent_class
        return decorator

    def __init__(self, **dependencies):
        self._instances = {}
        self._dependencies = dependencies

    def get(self, agent_name: str):
        agent_name = agent_name.lower()
        if agent_name not in self._registry:
            raise ValueError(f"Agent '{agent_name}' not found in registry.")
        if agent_name not in self._instances:
            agent_class = self._registry[agent_name]["class"]
            # Filter dependencies to match the agent's constructor
            sig = signature(agent_class.__init__)
            accepted_args = list(sig.parameters.keys())[1:]  # skip 'self'
            filtered_deps = {k: v for k, v in self._dependencies.items() if k in accepted_args}
            self._instances[agent_name] = agent_class(**filtered_deps)
        return self._instances[agent_name]

    def get_capabilities(self) -> str:
        lines = ["Available Agents and their required inputs:"]
        for name, info in self._registry.items():
            lines.append(f"\n- {name.title()}: {info['capabilities']}")
            lines.append(f"  Inputs: {info['required_inputs']}")
        return "\n".join(lines)