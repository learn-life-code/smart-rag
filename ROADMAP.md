# Roadmap

Tracked future work (not yet built).

## Binaries — deepen the codegraph/binary path (LATER)

Smart RAG's source side now has a real call graph (Python AST + optional
tree-sitter). The **binary** side is the next frontier — for firmware/ELF/exe/build
artifacts where you can't read source:

- **capstone** — disassemble and recover **call edges from machine code**
  (`call`/`bl`/`jal` targets), so "what calls function X" works on a stripped
  binary, not just symbol tables.
- **DWARF debug info** (via `pyelftools`, already a dependency) — **source ↔ binary
  mapping**: which source file/line a symbol came from. The world-standard for
  attributing binary symbols back to code.
- Goal: a `binary_callgraph` adapter (or an enhancement to the existing codegraph
  tool) that emits the same `calls`/`defined_in` relation facts as the source call
  graph — so an agent asks the same question over source OR firmware.

Division of labor stays: codegraph = binary specialist; Smart RAG = source +
content + the unified query surface that absorbs both.

## Deferred from the codegraph-gaps work

- **Import/reference resolution** (codegraph-main Phase 3): resolve `imports M`
  edges to the actual defined module across files, so cross-file caller/callee
  chains connect fully. Edges exist; resolution to concrete targets is the gap.
- **Bundle the full binary extractor**: Smart RAG now extracts ELF symbols + strings
  itself (unstripped .so/.elf/.o). STRIPPED ELFs + proprietary firmware (Qualcomm
  MBN, sparse Android .img) still need a specialized extractor — Smart RAG detects
  these and advises running codegraph. A future built-in MBN/sparse parser would
  close this, but those are proprietary container formats.

## Other

- tree-sitter call graph: validate edge quality per language (C/C++/Java/Go/Rust)
  against real repos; current backend is generic AST-walk, may need per-language
  query refinement.
- SSH collector: first real-machine run + a `--dry-run` that prints the read-only
  command set before connecting.
