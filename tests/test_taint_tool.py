import asyncio
import subprocess
from pathlib import Path

import lief
import pytest

from triton_taint_mcp.main import taint_binary


def build_asm(tmp_path: Path, name: str, source: str, pie: bool = False) -> Path:
    asm = tmp_path / f"{name}.s"
    asm.write_text(source)
    bin_path = tmp_path / name
    flags = ["gcc", "-nostdlib"]
    flags += ["-Wl,-pie"] if pie else ["-static", "-no-pie"]
    subprocess.run(
        [*flags, "-o", str(bin_path), str(asm)],
        check=True,
    )
    return bin_path


def symbol_addr(binary: lief.Binary, name: str) -> int:
    for sym in binary.symbols:
        if sym.name == name:
            return sym.value
    raise KeyError(f"symbol {name} not found")


def first_hlt_addr(binary: lief.Binary) -> int:
    text = binary.get_section(".text")
    content = list(text.content)
    try:
        idx = content.index(0xF4)  # hlt opcode
    except ValueError:
        raise AssertionError("no hlt found in .text")
    return text.virtual_address + idx


@pytest.mark.parametrize("taint_size", [8, 4])
def test_memory_to_memory_taint(tmp_path: Path, taint_size: int):
    """
    Taint data at src, ensure it propagates through load/add/store into dst.
    """
    src = r"""
    .intel_syntax noprefix
    .section .data
    .globl src
    .globl dst
    src: .quad 0x1122334455667788
    dst: .quad 0
    .section .text
    .globl _start
    _start:
        lea rdi, [rip+src]
        lea rsi, [rip+dst]
        mov rax, [rdi]
        add rax, 1
        mov [rsi], rax
        hlt
    """
    bin_path = build_asm(tmp_path, "mem2mem", src)
    binary = lief.parse(str(bin_path))
    src_addr = symbol_addr(binary, "src")
    dst_addr = symbol_addr(binary, "dst")
    hlt_addr = first_hlt_addr(binary)

    report = asyncio.run(
        taint_binary(
            binary_path=str(bin_path),
            taint_memory=[{"address": src_addr, "size": taint_size}],
            max_instructions=20,
            stop_addresses=[hlt_addr],
        )
    )

    tainted_mem_writes = [
        rec
        for rec in report["tainted_instructions"]
        if any(addr == dst_addr for addr, _ in rec["tainted_memory"])
    ]
    assert tainted_mem_writes, "taint should reach dst memory"


def test_register_taint_to_memory(tmp_path: Path):
    """
    Taint register rdi and ensure storing it taints memory.
    """
    src = r"""
    .intel_syntax noprefix
    .section .bss
    .balign 8
    .globl dst
    dst: .quad 0
    .section .text
    .globl _start
    _start:
        lea rax, [rip+dst]
        mov [rax], rdi
        hlt
    """
    bin_path = build_asm(tmp_path, "reg2mem", src)
    binary = lief.parse(str(bin_path))
    dst_addr = symbol_addr(binary, "dst")
    hlt_addr = first_hlt_addr(binary)

    report = asyncio.run(
        taint_binary(
            binary_path=str(bin_path),
            taint_registers=["rdi"],
            initial_registers={"rdi": 0x41},
            max_instructions=10,
            stop_addresses=[hlt_addr],
        )
    )

    tainted_dst = [
        rec
        for rec in report["tainted_instructions"]
        if any(addr == dst_addr for addr, _ in rec["tainted_memory"])
    ]
    assert tainted_dst, "tainted rdi should propagate to dst memory"


def test_no_taint_without_sources(tmp_path: Path):
    src = r"""
    .intel_syntax noprefix
    .section .text
    .globl _start
    _start:
        mov eax, 0
        hlt
    """
    bin_path = build_asm(tmp_path, "no_taint", src)
    binary = lief.parse(str(bin_path))
    hlt_addr = first_hlt_addr(binary)

    report = asyncio.run(
        taint_binary(
            binary_path=str(bin_path),
            max_instructions=5,
            stop_addresses=[hlt_addr],
        )
    )
    assert report["tainted_instructions"] == []


def test_pointer_deref_chain_taint(tmp_path: Path):
    """
    Taint memory reachable via a pointer stored in .data, ensure deref + store taints dst.
    """
    src = r"""
    .intel_syntax noprefix
    .section .data
    .globl ptr
    .globl src
    .globl dst
    ptr: .quad src
    src: .quad 0x4141414141414141
    dst: .quad 0
    .section .text
    .globl _start
    _start:
        lea rbx, [rip+ptr]
        mov rbx, [rbx]      # rbx = &src
        mov rax, [rbx]      # load tainted data
        mov [rip+dst], rax  # store to dst
        hlt
    """
    bin_path = build_asm(tmp_path, "ptr_taint", src)
    binary = lief.parse(str(bin_path))
    src_addr = symbol_addr(binary, "src")
    dst_addr = symbol_addr(binary, "dst")
    hlt_addr = first_hlt_addr(binary)

    report = asyncio.run(
        taint_binary(
            binary_path=str(bin_path),
            taint_memory=[{"address": src_addr, "size": 8}],
            max_instructions=20,
            stop_addresses=[hlt_addr],
        )
    )

    tainted_dst = [
        rec
        for rec in report["tainted_instructions"]
        if any(addr == dst_addr for addr, _ in rec["tainted_memory"])
    ]
    assert tainted_dst, "taint through pointer chain should reach dst"


def test_taint_cleared_after_overwrite(tmp_path: Path):
    """
    Tainted memory loaded then overwritten with zero should not leave tainted dst.
    """
    src = r"""
    .intel_syntax noprefix
    .section .data
    .globl src
    .globl dst
    src: .quad 0x1122334455667788
    dst: .quad 0
    .section .text
    .globl _start
    _start:
        lea rdi, [rip+src]
        mov rax, [rdi]   # rax tainted
        xor rax, rax     # clear taint
        mov [rip+dst], rax
        hlt
    """
    bin_path = build_asm(tmp_path, "clear_taint", src)
    binary = lief.parse(str(bin_path))
    src_addr = symbol_addr(binary, "src")
    dst_addr = symbol_addr(binary, "dst")
    hlt_addr = first_hlt_addr(binary)

    report = asyncio.run(
        taint_binary(
            binary_path=str(bin_path),
            taint_memory=[{"address": src_addr, "size": 8}],
            max_instructions=20,
            stop_addresses=[hlt_addr],
        )
    )

    tainted_dst = [
        rec
        for rec in report["tainted_instructions"]
        if any(addr == dst_addr for addr, _ in rec["tainted_memory"])
    ]
    assert not tainted_dst, "cleared register should not taint dst"


def test_initial_register_and_concrete_memory(tmp_path: Path):
    """
    Prefill memory via concrete_memory and seed register to point to it; taint the bytes and see propagation.
    """
    src = r"""
    .intel_syntax noprefix
    .section .data
    .globl dst
    dst: .quad 0
    .section .text
    .globl _start
    _start:
        mov rax, [rdi]        # rdi points to user buffer
        mov [rip+dst], rax
        hlt
    """
    bin_path = build_asm(tmp_path, "seed_state", src)
    binary = lief.parse(str(bin_path))
    dst_addr = symbol_addr(binary, "dst")
    hlt_addr = first_hlt_addr(binary)
    user_buf = 0x500000  # arbitrary mapped slot not used by segments

    report = asyncio.run(
        taint_binary(
            binary_path=str(bin_path),
            initial_registers={"rdi": user_buf},
            concrete_memory=[{"address": user_buf, "data": [0x41] * 8}],
            taint_memory=[{"address": user_buf, "size": 8}],
            max_instructions=10,
            stop_addresses=[hlt_addr],
        )
    )

    tainted_dst = [
        rec
        for rec in report["tainted_instructions"]
        if any(addr == dst_addr for addr, _ in rec["tainted_memory"])
    ]
    assert tainted_dst, "taint from user buffer via seeded pointer should reach dst"


def test_stop_address_halts(tmp_path: Path):
    """
    Ensure stop_addresses halts execution early.
    """
    src = r"""
    .intel_syntax noprefix
    .section .text
    .globl _start
    _start:
        add eax, 1
        jmp _start
    """
    bin_path = build_asm(tmp_path, "stopaddr", src)
    binary = lief.parse(str(bin_path))
    stop_addr = binary.entrypoint  # halt immediately

    report = asyncio.run(
        taint_binary(
            binary_path=str(bin_path),
            stop_addresses=[stop_addr],
            max_instructions=100,
        )
    )
    assert report["stopped_on"] == f"hit stop address 0x{stop_addr:x}"
    assert report["instruction_count"] == 0


def test_pie_base_added(tmp_path: Path):
    """
    PIE binaries load at base 0x00100000. Verify addresses align for tainting.
    """
    src = r"""
    .intel_syntax noprefix
    .section .data
    .globl src
    src: .quad 0x1337
    .section .text
    .globl _start
    _start:
        lea rdi, [rip+src]
        mov rax, [rdi]
        hlt
    """
    bin_path = build_asm(tmp_path, "pie_bin", src, pie=True)
    binary = lief.parse(str(bin_path))
    src_vaddr = symbol_addr(binary, "src")
    base = 0x00100000
    src_runtime = base + src_vaddr
    hlt_addr = first_hlt_addr(binary) + base

    report = asyncio.run(
        taint_binary(
            binary_path=str(bin_path),
            taint_memory=[{"address": src_runtime, "size": 8}],
            max_instructions=20,
            stop_addresses=[hlt_addr],
        )
    )

    tainted = any(rec["tainted"] for rec in report["tainted_instructions"])
    assert tainted, "PIE load with forced base should still taint"
