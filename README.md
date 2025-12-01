## Triton Taint MCP

Model Context Protocol (MCP) stdio server that exposes a single tool, `taint_binary`, backed by the [Triton](https://github.com/JonathanSalwan/Triton) dynamic symbolic execution engine. It provides a lightweight forward-taint pass suitable for LLM-driven vulnerability triage: given a binary, initial taint sources (registers and/or memory), and a step budget, it reports which instructions propagate user-controlled data into registers or memory.

### Quick start

```
uv sync          # installs mcp, lief, triton-library (PyPI), pydantic
uv run triton-taint-mcp
```

The server speaks MCP over stdio; point your MCP-compatible client at the executable `triton-taint-mcp` (exported by the `pyproject.toml` script entry).

### Tool interface

`taint_binary` parameters:
- `binary_path` (str, required): Path to ELF/PE to load via LIEF and map into Triton.
- `arch` (str, default `x86_64`): One of `x86_64`, `x86`, `aarch64`.
- `entrypoint` (int|hex-string, optional): Address to start execution (defaults to binary entrypoint).
- `taint_registers` (list[str], optional): Register names to taint initially, e.g. `["rdi", "rsi"]`.
- `taint_memory` (list[{"address": int|hex-string, "size": int|hex-string}], optional): Memory ranges to taint initially.
- `initial_registers` (dict[str, int|hex-string], optional): Seed concrete register values (e.g. set pointers for memory loads).
- `concrete_memory` (list[{"address": int|hex-string, "data": list[int]|bytes|hex-string}], optional): Prefill memory at arbitrary addresses; hex strings are accepted.
- `max_instructions` (int, default 500): Step budget to avoid runaway emulation.
- `stop_addresses` (list[int|hex-string], optional): Addresses that halt emulation when reached.

Return value:
```
{
  "instruction_count": <int>,         # count of tainted instruction records
  "tainted_instructions": [
    {
      "address": <int>,
      "mnemonic": "0x401000: mov eax, [rbp-4]",
      "tainted": <bool>,               # instruction reads or writes tainted data
      "tainted_registers": ["eax"],    # registers that are tainted after this instruction
      "tainted_memory": [[0x7ffff000, 4]]
    },
    ...
  ],
  "stopped_on": "<reason>|null",
  "error": "<message>|null"
}
```

### How it works (brief)
1. LIEF maps loadable segments into Triton memory.
2. Basic stack is seeded (RSP/RBP) to keep execution sane.
3. Taint sources are applied to requested registers/memory.
4. Linear stepping: fetches up to 16 bytes at `rip`, decodes/executes via Triton, follows the updated `rip`.
5. Each step records writes that end up tainted (registers or memory) and any instruction marked tainted by Triton.

This is intentionally minimal—no libc hooking, SMT solving, or relocation fixes beyond segment mapping. It’s designed for quick directional answers (“does user-controlled data reach this compare/store within N steps?”). Pair it with a richer disassembler MCP for context and with your own stopping conditions to constrain exploration.

### Notes and caveats
- Requires Python ≥3.11 and a working compiler toolchain to build Triton; it is installed from PyPI as `triton-library`.
- LIEF handles ELF/PE parsing; Mach-O isn’t wired up.
- Execution is single-threaded, linear, and stops at `max_instructions` or `stop_addresses`. Indirect control flows that loop may be truncated.
- If you want to experiment against the local Triton clone, you can uncomment the `triton-local` editable source in `pyproject.toml` to override the PyPI package.
- PIE ELF binaries are mapped at base address `0x00100000` (segments shifted by that base; entrypoint likewise). Non-PIE binaries are mapped at their recorded VAs. Provide addresses accordingly (runtime addresses for PIE = base + ELF VA).
