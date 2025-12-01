from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from mcp import types
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server

from .taint import TaintSource, simple_forward_taint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("triton-taint-mcp", version="0.1.0")


def _to_int(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        # accepts "0x...", decimal strings, or plain hex without 0x via base=0
        return int(value, 0)
    raise TypeError(f"Expected int or str for address/size, got {type(value)}")


async def taint_binary(
    binary_path: str,
    arch: str = "x86_64",
    entrypoint: Optional[int] = None,
    taint_registers: Optional[List[str]] = None,
    taint_memory: Optional[List[Dict[str, int]]] = None,
    initial_registers: Optional[Dict[str, int]] = None,
    concrete_memory: Optional[List[Dict]] = None,
    max_instructions: int = 500,
    stop_addresses: Optional[List[int]] = None,
) -> Dict:
    """
    Run a lightweight forward-taint pass over a binary using Triton.

    Parameters
    ----------
    binary_path : path to the ELF/PE to map into Triton.
    arch : one of x86_64, x86, aarch64 (default x86_64).
    entrypoint : address to start from (defaults to binary entrypoint).
    taint_registers : list of register names to taint at start (e.g. ['rdi','rsi']).
    taint_memory : list of dicts {\"address\": int, \"size\": int} to taint initial memory ranges.
    initial_registers : dict of register -> value to seed concrete state (e.g. {'rdi': 0x600000}).
    concrete_memory : list of dicts {\"address\": int, \"data\": list[int]|bytes} to prefill memory.
    max_instructions : hard stop to prevent runaway emulation.
    stop_addresses : list of addresses that will stop emulation if hit.
    """
    sources: List[TaintSource] = []
    for reg in taint_registers or []:
        sources.append(TaintSource(kind="register", target=reg))
    for mem in taint_memory or []:
        sources.append(
            TaintSource(
                kind="memory",
                target=_to_int(mem["address"]),
                size=_to_int(mem.get("size", 1)),
            )
        )

    initial_registers = {k: _to_int(v) for k, v in (initial_registers or {}).items()}

    parsed_concrete_memory = []
    for mem in concrete_memory or []:
        data = mem["data"]
        if isinstance(data, str):
            # interpret as hexstring (with or without 0x); if odd length, pad
            hex_str = data[2:] if data.startswith("0x") else data
            if len(hex_str) % 2 == 1:
                hex_str = "0" + hex_str
            data_bytes = bytes.fromhex(hex_str)
            data_list = list(data_bytes)
        elif isinstance(data, (bytes, bytearray)):
            data_list = list(data)
        else:
            data_list = [int(b) & 0xFF for b in data]
        parsed_concrete_memory.append(
            {"address": _to_int(mem["address"]), "data": data_list}
        )

    report = simple_forward_taint(
        binary_path=binary_path,
        arch=arch,
        entrypoint=_to_int(entrypoint),
        taint_sources=sources,
        initial_registers=initial_registers,
        concrete_memory=parsed_concrete_memory,
        max_instructions=max_instructions,
        stop_addresses=[_to_int(a) for a in (stop_addresses or [])],
    )

    return report.to_dict()


# MCP wiring (list/call tools)


@server.list_tools()
async def list_tools() -> List[types.Tool]:
    return [
        types.Tool(
            name="taint_binary",
            description="Run a lightweight Triton forward-taint analysis over a binary.",
            inputSchema={
                "type": "object",
                "properties": {
                    "binary_path": {"type": "string"},
                    "arch": {"type": "string", "enum": ["x86_64", "x86", "aarch64"]},
                    "entrypoint": {"type": ["integer", "string", "null"]},
                    "taint_registers": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "taint_memory": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "address": {"type": ["integer", "string"]},
                                "size": {"type": ["integer", "string"]},
                            },
                            "required": ["address"],
                        },
                    },
                    "initial_registers": {
                        "type": "object",
                        "additionalProperties": {"type": ["integer", "string"]},
                    },
                    "concrete_memory": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "address": {"type": ["integer", "string"]},
                                "data": {
                                    "oneOf": [
                                        {"type": "array", "items": {"type": "integer"}},
                                        {"type": "string"},
                                    ]
                                },
                            },
                            "required": ["address", "data"],
                        },
                    },
                    "max_instructions": {"type": "integer"},
                    "stop_addresses": {
                        "type": "array",
                        "items": {"type": ["integer", "string"]},
                    },
                },
                "required": ["binary_path"],
                "additionalProperties": False,
            },
            outputSchema=None,
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict):
    if name != "taint_binary":
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Unknown tool {name}")],
            isError=True,
        )
    result = await taint_binary(**arguments)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text="ok")],
        structuredContent=result,
        isError=False,
    )


def run():
    async def _main():
        init_opts = server.create_initialization_options(
            notification_options=NotificationOptions(),
        )
        async with stdio_server() as (read, write):
            await server.run(
                read_stream=read,
                write_stream=write,
                initialization_options=init_opts,
            )

    asyncio.run(_main())


if __name__ == "__main__":
    run()
