#!/usr/bin/env python3
"""Generate SQLite opcodes.h without requiring Tcl."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: mkopcodeh.py PARSE_H VDBE_C OUT_H", file=sys.stderr)
        return 2

    parse_h = Path(sys.argv[1])
    vdbe_c = Path(sys.argv[2])
    out_h = Path(sys.argv[3])

    tk: dict[str, int] = {}
    op: dict[str, int] = {}
    used: dict[int, bool] = {}
    sameas: dict[int, str] = {}
    definition: dict[int, str] = {}
    group: dict[str, bool] = {}
    jump: dict[str, bool] = {}
    jump0: dict[str, bool] = {}
    in1: dict[str, bool] = {}
    in2: dict[str, bool] = {}
    in3: dict[str, bool] = {}
    out2: dict[str, bool] = {}
    out3: dict[str, bool] = {}
    ncycle: dict[str, bool] = {}
    synopsis: dict[str, str] = {}
    groups: list[list[str]] = []
    order: list[str] = []
    current_op = ""
    prev_name = ""
    n_group = 0

    def ensure_group(index: int) -> list[str]:
        while len(groups) <= index:
            groups.append([])
        return groups[index]

    for path in (parse_h, vdbe_c):
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.rstrip("\n")
                if line.startswith("#define TK_"):
                    parts = line.split()
                    if len(parts) >= 3:
                        tk[parts[1]] = int(parts[2])
                    continue

                if re.match(r"^.. Opcode: ", line):
                    parts = line.split()
                    if len(parts) >= 3:
                        current_op = "OP_" + parts[2]
                        # Kept for parity with mkopcodeh.tcl; currently not emitted.
                        _ = sum(
                            bit
                            for term, bit in {
                                "P1": 1,
                                "P2": 2,
                                "P3": 4,
                                "P4": 8,
                                "P5": 16,
                            }.items()
                            if term in parts
                        )
                    continue

                match = re.match(r"^.. Synopsis: (.*)", line)
                if match and current_op:
                    synopsis[current_op] = match.group(1).strip()
                    continue

                if not line.startswith("case OP_"):
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue
                name = parts[1].rstrip(":")
                if name == "OP_Abortable":
                    continue

                op[name] = -1
                group[name] = False
                jump[name] = False
                jump0[name] = False
                in1[name] = False
                in2[name] = False
                in3[name] = False
                out2[name] = False
                out3[name] = False
                ncycle[name] = False

                i = 3
                while i < len(parts) - 1:
                    term = parts[i].strip(",")
                    if term == "same" and i + 2 < len(parts) and parts[i + 1] == "as":
                        sym = parts[i + 2].strip(",")
                        val = tk[sym]
                        op[name] = val
                        used[val] = True
                        sameas[val] = sym
                        definition[val] = name
                        i += 3
                        continue
                    if term == "group":
                        group[name] = True
                    elif term == "jump":
                        jump[name] = True
                    elif term == "in1":
                        in1[name] = True
                    elif term == "in2":
                        in2[name] = True
                    elif term == "in3":
                        in3[name] = True
                    elif term == "out2":
                        out2[name] = True
                    elif term == "out3":
                        out3[name] = True
                    elif term == "ncycle":
                        ncycle[name] = True
                    elif term == "jump0":
                        jump[name] = True
                        jump0[name] = True
                    i += 1

                if group[name]:
                    new_group = False
                    if n_group < len(groups) and (not prev_name or not group.get(prev_name, False)):
                        new_group = True
                    ensure_group(n_group).append(name)
                    if new_group:
                        n_group += 1
                elif prev_name and group.get(prev_name, False):
                    n_group += 1

                order.append(name)
                prev_name = name

    for name in ("OP_Noop", "OP_Explain", "OP_Abortable"):
        jump[name] = jump0[name] = in1[name] = in2[name] = False
        in3[name] = out2[name] = out3[name] = ncycle[name] = False
        op[name] = -1
        order.append(name)

    rp2v_ops = {
        "OP_Transaction",
        "OP_AutoCommit",
        "OP_Savepoint",
        "OP_Checkpoint",
        "OP_Vacuum",
        "OP_JournalMode",
        "OP_VUpdate",
        "OP_VFilter",
        "OP_Init",
    }

    cnt = -1
    for name in order:
        if name in rp2v_ops:
            cnt += 1
            while cnt in used:
                cnt += 1
            op[name] = cnt
            used[cnt] = True
            definition[cnt] = name

    for name in order:
        if op[name] >= 0 or not jump[name]:
            continue
        cnt += 1
        while cnt in used:
            cnt += 1
        op[name] = cnt
        used[cnt] = True
        definition[cnt] = name

    mx_jump = -1
    for name in order:
        if jump[name] and op[name] > mx_jump:
            mx_jump = op[name]

    for members in groups:
        if not members:
            continue
        seek = cnt
        while True:
            seek += 1
            while seek in used:
                seek += 1
            start = seek
            ok = True
            probe = seek
            for _ in members:
                probe += 1
                if probe in used:
                    ok = False
                    break
            if ok:
                next_value = start
                for name in members:
                    if op[name] < 0:
                        op[name] = next_value
                        used[next_value] = True
                        definition[next_value] = name
                        next_value += 1
                break

    for name in order:
        if op[name] < 0:
            cnt += 1
            while cnt in used:
                cnt += 1
            op[name] = cnt
            used[cnt] = True
            definition[cnt] = name

    max_opcode = max(used)
    if max_opcode > 255:
        raise RuntimeError("More than 255 opcodes - VdbeOp.opcode is of type u8!")

    lines: list[str] = [
        "/* Automatically generated.  Do not edit */",
        "/* See the tool/mkopcodeh.py script for details */",
    ]
    for value in range(max_opcode + 1):
        name = definition.get(value, f"OP_NotUsed_{value}")
        comment: list[str] = []
        if jump0.get(name, False):
            comment.append("jump0")
        elif jump.get(name, False):
            comment.append("jump")
        if value in sameas:
            comment.append(f"same as {sameas[value]}")
        if name in synopsis:
            comment.append(f"synopsis: {synopsis[name]}")
        line = f"#define {name:<16} {value:3d}"
        if comment:
            line += f" /* {', '.join(comment):<42} */"
        lines.append(line)

    bitvectors: list[int] = []
    for value in range(max_opcode + 1):
        name = definition.get(value, f"OP_NotUsed_{value}")
        bits = 0
        if not name.startswith("OP_NotUsed"):
            bits |= 0x01 if jump.get(name, False) else 0
            bits |= 0x02 if in1.get(name, False) else 0
            bits |= 0x04 if in2.get(name, False) else 0
            bits |= 0x08 if in3.get(name, False) else 0
            bits |= 0x10 if out2.get(name, False) else 0
            bits |= 0x20 if out3.get(name, False) else 0
            bits |= 0x40 if ncycle.get(name, False) else 0
            bits |= 0x80 if jump0.get(name, False) else 0
        bitvectors.append(bits)

    lines.extend(
        [
            "",
            '/* Properties such as "out2" or "jump" that are specified in',
            '** comments following the "case" for each opcode in the vdbe.c',
            "** are encoded into bitvectors as follows:",
            "*/",
            "#define OPFLG_JUMP        0x01  /* jump:  P2 holds jmp target */",
            "#define OPFLG_IN1         0x02  /* in1:   P1 is an input */",
            "#define OPFLG_IN2         0x04  /* in2:   P2 is an input */",
            "#define OPFLG_IN3         0x08  /* in3:   P3 is an input */",
            "#define OPFLG_OUT2        0x10  /* out2:  P2 is an output */",
            "#define OPFLG_OUT3        0x20  /* out3:  P3 is an output */",
            "#define OPFLG_NCYCLE      0x40  /* ncycle:Cycles count against P1 */",
            "#define OPFLG_JUMP0       0x80  /* jump0:  P2 might be zero */",
            "#define OPFLG_INITIALIZER {\\",
        ]
    )
    for start in range(0, len(bitvectors), 8):
        chunk = bitvectors[start : start + 8]
        values = "".join(f" 0x{value:02x}," for value in chunk)
        lines.append(f"/* {start:3d} */{values}\\")
    lines.extend(
        [
            "}",
            "",
            "/* The resolve3P2Values() routine is able to run faster if it knows",
            "** the value of the largest JUMP opcode.  The smaller the maximum",
            "** JUMP opcode the better, so this generator groups all JUMP opcodes",
            "** together near the beginning of the list.",
            "*/",
            f"#define SQLITE_MX_JUMP_OPCODE  {mx_jump}  /* Maximum JUMP opcode */",
            "",
        ]
    )

    out_h.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
