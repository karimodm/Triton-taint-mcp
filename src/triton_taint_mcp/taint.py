from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Sequence, Tuple

import lief
from triton import ARCH, Instruction, MemoryAccess, TritonContext

logger = logging.getLogger(__name__)


ARCH_MAP = {
    "x86_64": ARCH.X86_64,
    "x86": ARCH.X86,
    "aarch64": ARCH.AARCH64,
}


@dataclass
class TaintSource:
    kind: str  # "register" or "memory"
    target: str | int  # register name or memory address
    size: int | None = None  # required for memory


@dataclass
class TaintedWrite:
    address: int
    mnemonic: str
    tainted: bool
    tainted_registers: List[str]
    tainted_memory: List[Tuple[int, int]]


@dataclass
class TaintReport:
    instruction_count: int
    tainted_instructions: List[TaintedWrite]
    stopped_on: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def _load_binary(ctx: TritonContext, path: str):
    """
    Map binary segments into Triton memory using lief.
    """
    binary = lief.parse(path)
    if binary is None:
        raise ValueError(f"Failed to parse binary at {path}")

    base = 0x00100000 if getattr(binary, "is_pie", False) else 0

    for segment in binary.segments:
        size = segment.physical_size
        vaddr = segment.virtual_address
        if size == 0:
            continue
        logger.debug("Mapping segment 0x%x - 0x%x", vaddr, vaddr + size)
        ctx.setConcreteMemoryAreaValue(base + vaddr, list(segment.content))
    return binary, base


def _apply_taint_sources(ctx: TritonContext, sources: Sequence[TaintSource]):
    for source in sources:
        if source.kind == "register":
            reg = getattr(ctx.registers, str(source.target))
            ctx.taintRegister(reg)
        elif source.kind == "memory":
            if source.size is None:
                raise ValueError("memory taint source requires size")
            ctx.taintMemory(MemoryAccess(int(source.target), int(source.size)))
        else:
            raise ValueError(f"Unsupported taint source kind: {source.kind}")


def _capture_tainted_writes(ctx: TritonContext, inst: Instruction) -> TaintedWrite:
    written_regs = []
    for reg_access in inst.getWrittenRegisters():
        reg = reg_access[0] if isinstance(reg_access, tuple) else reg_access
        if ctx.isRegisterTainted(reg):
            written_regs.append(reg.getName())

    written_mem = []
    for mem_access in inst.getStoreAccess():
        mem = mem_access[0] if isinstance(mem_access, tuple) else mem_access
        if ctx.isMemoryTainted(mem):
            written_mem.append((mem.getAddress(), mem.getSize()))
    return TaintedWrite(
        address=inst.getAddress(),
        mnemonic=str(inst),
        tainted=inst.isTainted(),
        tainted_registers=written_regs,
        tainted_memory=written_mem,
    )


def simple_forward_taint(
    binary_path: str,
    arch: str = "x86_64",
    entrypoint: Optional[int] = None,
    taint_sources: Optional[Sequence[TaintSource]] = None,
    initial_registers: Optional[dict[str, int]] = None,
    concrete_memory: Optional[Sequence[dict]] = None,
    max_instructions: int = 500,
    stop_addresses: Optional[Iterable[int]] = None,
    stack_base: int = 0x7FFFFFF0,
) -> TaintReport:
    """
    Lightweight forward-taint pass. It loads an ELF/PE using LIEF,
    maps segments, applies user taints, then single-steps instructions,
    following rip as set by Triton.
    """
    ctx = TritonContext(ARCH_MAP[arch])
    binary, base = _load_binary(ctx, binary_path)

    # Set a basic stack so rip has space to grow.
    ctx.setConcreteRegisterValue(ctx.registers.rbp, stack_base)
    ctx.setConcreteRegisterValue(ctx.registers.rsp, stack_base)

    for reg_name, value in (initial_registers or {}).items():
        reg = getattr(ctx.registers, reg_name)
        ctx.setConcreteRegisterValue(reg, int(value))

    for mem in concrete_memory or []:
        addr = int(mem["address"])
        data = mem["data"]
        if isinstance(data, (bytes, bytearray)):
            values = list(data)
        else:
            values = [int(b) & 0xFF for b in data]
        ctx.setConcreteMemoryAreaValue(addr, values)

    if taint_sources:
        _apply_taint_sources(ctx, taint_sources)

    pc = entrypoint or int(base + binary.entrypoint)
    tainted: List[TaintedWrite] = []
    stop_set = set(stop_addresses or [])
    stopped_on: Optional[str] = None

    ctx.setConcreteRegisterValue(ctx.registers.rip, pc)

    for idx in range(max_instructions):
        try:
            opcode = bytes(ctx.getConcreteMemoryAreaValue(pc, 16))
        except Exception as exc:
            stopped_on = f"memory fetch failed at 0x{pc:x}: {exc}"
            break

        inst = Instruction()
        inst.setOpcode(opcode)
        inst.setAddress(pc)

        try:
            ctx.processing(inst)
        except Exception as exc:
            stopped_on = f"Triton processing failed at 0x{pc:x}: {exc}"
            break

        rec = _capture_tainted_writes(ctx, inst)
        if rec.tainted or rec.tainted_registers or rec.tainted_memory:
            tainted.append(rec)

        pc = ctx.getConcreteRegisterValue(ctx.registers.rip)
        if pc in stop_set:
            stopped_on = f"hit stop address 0x{pc:x}"
            break

    return TaintReport(
        instruction_count=len(tainted),
        tainted_instructions=tainted,
        stopped_on=stopped_on,
    )
