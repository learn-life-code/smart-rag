#!/usr/bin/env python3
"""Smart RAG GUI — drop a file/folder, ask your data, get grounded source-cited answers.

For non-coders: no CLI needed. Pick files → Ingest → type a question → see the
answer with its sources. Also a 'Search' box for exact keyword/entity lookups.

    py -3.13 distill/gui.py
"""
import os
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smart_rag import SmartRAG   # noqa: E402


class DistillGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart RAG — ask your data")
        self.geometry("900x680")
        self.distiller = SmartRAG()
        self._build()

    def _build(self):
        top = ttk.Frame(self, padding=10); top.pack(fill="x")
        ttk.Label(top, text="Smart RAG", font=("Segoe UI", 16, "bold")).pack(side="left")
        ttk.Label(top, text="  point it at your files/logs → ask → grounded answers",
                  foreground="#6b7280").pack(side="left")

        bar = ttk.Frame(self, padding=(10, 0)); bar.pack(fill="x")
        ttk.Button(bar, text="📄 Add files…", command=self._add_files).pack(side="left", padx=(0, 6))
        ttk.Button(bar, text="📁 Add folder…", command=self._add_folder).pack(side="left", padx=(0, 6))
        ttk.Button(bar, text="🗑 Clear", command=self._clear).pack(side="left")
        self._stat = tk.StringVar(value="No data ingested yet.")
        ttk.Label(bar, textvariable=self._stat, foreground="#166534").pack(side="left", padx=12)

        # Ask
        qf = ttk.LabelFrame(self, text="Ask (grounded answer)", padding=8); qf.pack(fill="x", padx=10, pady=8)
        self._q = tk.StringVar()
        e = ttk.Entry(qf, textvariable=self._q, width=70); e.pack(side="left", fill="x", expand=True)
        e.bind("<Return>", lambda _ : self._ask())
        ttk.Button(qf, text="Ask", command=self._ask).pack(side="left", padx=6)
        # AI toggle: when on, ALSO show an AI-summarized version (uses an optional
        # BYO LLM — see SmartRAG.llm). Grounded answer is always shown first.
        self._use_ai = tk.BooleanVar(value=False)
        ttk.Checkbutton(qf, text="+ AI summary", variable=self._use_ai).pack(side="left", padx=4)
        self._llm = None

        # Search
        sf = ttk.LabelFrame(self, text="Search (exact, no AI)", padding=8); sf.pack(fill="x", padx=10)
        ttk.Label(sf, text="entity:").pack(side="left"); self._se = tk.StringVar()
        ttk.Entry(sf, textvariable=self._se, width=18).pack(side="left", padx=(2, 8))
        ttk.Label(sf, text="attribute:").pack(side="left"); self._sa = tk.StringVar()
        ttk.Entry(sf, textvariable=self._sa, width=18).pack(side="left", padx=(2, 8))
        ttk.Button(sf, text="Search", command=self._search).pack(side="left")

        self._out = scrolledtext.ScrolledText(self, wrap="word", font=("Consolas", 10))
        self._out.pack(fill="both", expand=True, padx=10, pady=10)

    # ── actions ──────────────────────────────────────────────────────────────
    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select files to ingest",
            filetypes=[("Data", "*.xlsx *.xlsm *.csv *.json *.log *.txt *.dlt"), ("All", "*.*")])
        for p in paths:
            self._ingest(p)

    def _add_folder(self):
        d = filedialog.askdirectory(title="Select a folder to ingest")
        if d:
            self._ingest(d)

    def _ingest(self, path):
        self._stat.set(f"Ingesting {os.path.basename(path)}…  (this can take a minute)")
        self._out.insert("end", f"\n⏳ Ingesting {path} …\n")
        self.update_idletasks()

        def work():
            # quiet ingest (no console spam); show a clean summary in the window
            st = self.distiller.ingest(path, verbose=False)
            msg = (f"✅ {st['files_ingested']} files → {st['entities']:,} entities · "
                   f"{st['distinct_facts']:,} facts · {st['prose_chunks']:,} prose chunks")
            if st.get("files_skipped_unsupported"):
                msg += f"  ({st['files_skipped_unsupported']} unsupported skipped)"
            self._stat.set(msg)
            self._out.insert("end", msg + "\n")
            self._out.see("end")
        threading.Thread(target=work, daemon=True).start()

    def _clear(self):
        self.distiller = SmartRAG()
        self._stat.set("Cleared.")
        self._out.delete("1.0", "end")

    def _ask(self):
        q = self._q.get().strip()
        if not q:
            return
        self._out.insert("end", f"\n❓ {q}\n")
        # Structured grounded answer FIRST (status + confidence + evidence).
        res = self.distiller.answer(q)
        self._out.insert("end", "── GROUNDED (no AI, source of truth) ──\n")
        self._out.insert("end", res.to_text() + "\n")
        self._out.see("end")
        # Optional AI summary of the SAME grounded evidence (toggle).
        if self._use_ai.get():
            self._out.insert("end", "\n── AI SUMMARY (phrasing of the above; grounded) ──\n⏳…\n")
            self._out.see("end")
            self.update_idletasks()
            threading.Thread(target=self._ai_summary, args=(q, res), daemon=True).start()

    def _ai_summary(self, query, res):
        llm = self._get_llm()
        if not llm:
            self._out.insert("end", "(no AI backend configured — set SmartRAG.llm to enable)\n")
            return
        grounding = res.to_text(show_evidence=True)
        try:
            txt = self.distiller._phrase(query, grounding, llm)
        except Exception as e:  # noqa: BLE001
            txt = f"(AI error: {e})"
        self._out.insert("end", (txt or "(AI returned nothing)") + "\n")
        self._out.see("end")

    def _get_llm(self):
        """Optional, BYO LLM for the 'AI summary' toggle. Smart RAG is grounded-only by
        default; the LLM merely PHRASES already-retrieved, cited facts (it never adds
        knowledge). Plug in your own by setting SmartRAG.llm to a callable
        ``fn(messages: list[dict]) -> str`` — e.g. an OpenAI/Anthropic/local wrapper.
        If none is set, the GUI stays grounded-only (the source of truth)."""
        return getattr(self.distiller, "llm", None)

    def _search(self):
        hits = self.distiller.search(entity=self._se.get() or None,
                                     attribute=self._sa.get() or None, limit=40)
        self._out.insert("end", f"\n🔎 search → {len(hits)} fact(s)\n")
        for h in hits:
            self._out.insert("end", f"  {h.entity} · {h.attribute}: {h.value}   [{h.source}]\n")
        self._out.see("end")


if __name__ == "__main__":
    DistillGUI().mainloop()
