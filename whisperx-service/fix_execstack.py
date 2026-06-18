"""Clear the executable-stack flag (PF_X) on PT_GNU_STACK in ELF shared libraries.

Some PyPI wheels (notably ctranslate2 4.4.0) ship native libraries marked as
requiring an executable stack. Hardened kernels (e.g. recent WSL2 kernels)
refuse to load such libraries and raise:

    ImportError: ... cannot enable executable stack as shared object requires

This script rewrites the GNU_STACK program-header flags in place so the
library no longer requests an executable stack. Pure Python — no external
dependencies (no execstack/prelink needed).
"""

import struct
import sys
from pathlib import Path

PT_GNU_STACK = 0x6474E551
PF_X = 0x1


def patch(path: Path) -> bool:
    data = bytearray(path.read_bytes())
    if data[:4] != b"\x7fELF":
        return False

    is_64 = data[4] == 2
    little = data[5] == 1
    endian = "<" if little else ">"

    if is_64:
        e_phoff = struct.unpack_from(endian + "Q", data, 32)[0]
        e_phentsize = struct.unpack_from(endian + "H", data, 54)[0]
        e_phnum = struct.unpack_from(endian + "H", data, 56)[0]
        flags_offset_in_ph = 4
    else:
        e_phoff = struct.unpack_from(endian + "I", data, 28)[0]
        e_phentsize = struct.unpack_from(endian + "H", data, 42)[0]
        e_phnum = struct.unpack_from(endian + "H", data, 44)[0]
        flags_offset_in_ph = 24

    changed = False
    for i in range(e_phnum):
        ph_off = e_phoff + i * e_phentsize
        p_type = struct.unpack_from(endian + "I", data, ph_off)[0]
        if p_type != PT_GNU_STACK:
            continue
        flags_off = ph_off + flags_offset_in_ph
        p_flags = struct.unpack_from(endian + "I", data, flags_off)[0]
        if p_flags & PF_X:
            new_flags = p_flags & ~PF_X
            struct.pack_into(endian + "I", data, flags_off, new_flags)
            changed = True
            print(f"  patched {path}: flags 0x{p_flags:x} -> 0x{new_flags:x}")
        break

    if changed:
        path.write_bytes(bytes(data))
    return changed


def main(roots):
    total = 0
    for root in roots:
        for p in Path(root).rglob("*.so*"):
            try:
                if patch(p):
                    total += 1
            except Exception as e:
                print(f"  skip {p}: {e}")
    print(f"[fix_execstack] patched {total} libraries")


if __name__ == "__main__":
    main(sys.argv[1:] or ["/usr/local/lib/python3.11/site-packages"])
