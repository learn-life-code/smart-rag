#!/usr/bin/env python3
"""Desktop GUI for persistent, source-cited SmartRAG stores."""
from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smart_rag import SmartRAG  # noqa: E402


def default_db_path() -> str:
    configured = os.environ.get("SMARTRAG_DB", "").strip()
    return str(Path(configured).expanduser() if configured
               else Path.home() / ".smartrag" / "smart_rag.db")


def format_store_status(distiller: SmartRAG, db_path: str) -> str:
    stats = distiller.store.stats()
    integrity = "ok"
    if distiller.db is not None:
        integrity = "ok" if distiller.db.integrity_ok() else "FAILED"
    return (
        f"{Path(db_path).name}: {stats['entities']:,} entities, "
        f"{stats['distinct_facts']:,} facts, "
        f"{stats['prose_chunks']:,} prose chunks; integrity {integrity}"
    )


class SmartRAGGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart RAG")
        self.geometry("980x720")
        self.minsize(760, 560)
        self._events: queue.Queue = queue.Queue()
        self._busy = False
        self._controls: list[ttk.Button] = []
        self._db_path = tk.StringVar(value=default_db_path())
        self._status = tk.StringVar(value="Opening database...")
        self._llm = None
        self._build()
        self.after(80, self._poll_events)
        self.after(120, self._open_database)

    def _build(self):
        top = ttk.Frame(self, padding=(12, 10))
        top.pack(fill="x")
        ttk.Label(top, text="Smart RAG",
                  font=("Segoe UI", 16, "bold")).pack(side="left")
        ttk.Label(top, text="  persistent, grounded, source-cited retrieval",
                  foreground="#6b7280").pack(side="left")

        db = ttk.LabelFrame(self, text="Database", padding=8)
        db.pack(fill="x", padx=12)
        ttk.Entry(db, textvariable=self._db_path).pack(
            side="left", fill="x", expand=True)
        self._button(db, "Browse", self._browse_database).pack(
            side="left", padx=(6, 2))
        self._button(db, "Open / Create", self._open_database).pack(side="left")

        actions = ttk.Frame(self, padding=(12, 8))
        actions.pack(fill="x")
        self._button(actions, "Add files", self._add_files).pack(
            side="left", padx=(0, 6))
        self._button(actions, "Add folder", self._add_folder).pack(
            side="left", padx=(0, 6))
        ttk.Button(actions, text="Clear output", command=self._clear_output).pack(
            side="left")
        ttk.Label(actions, textvariable=self._status,
                  foreground="#166534").pack(side="left", padx=12)

        ask = ttk.LabelFrame(self, text="Ask", padding=8)
        ask.pack(fill="x", padx=12, pady=(0, 8))
        self._question = tk.StringVar()
        question_entry = ttk.Entry(ask, textvariable=self._question)
        question_entry.pack(side="left", fill="x", expand=True)
        question_entry.bind("<Return>", lambda _event: self._ask())
        self._button(ask, "Ask", self._ask).pack(side="left", padx=6)
        self._use_ai = tk.BooleanVar(value=False)
        ttk.Checkbutton(ask, text="AI summary",
                        variable=self._use_ai).pack(side="left")

        search = ttk.LabelFrame(self, text="Exact fact search", padding=8)
        search.pack(fill="x", padx=12)
        ttk.Label(search, text="Entity").pack(side="left")
        self._entity = tk.StringVar()
        ttk.Entry(search, textvariable=self._entity, width=24).pack(
            side="left", padx=(4, 10))
        ttk.Label(search, text="Attribute").pack(side="left")
        self._attribute = tk.StringVar()
        ttk.Entry(search, textvariable=self._attribute, width=24).pack(
            side="left", padx=(4, 10))
        self._button(search, "Search", self._search).pack(side="left")

        self._output = scrolledtext.ScrolledText(
            self, wrap="word", font=("Consolas", 10), state="disabled")
        self._output.pack(fill="both", expand=True, padx=12, pady=12)

    def _button(self, parent, text, command) -> ttk.Button:
        button = ttk.Button(parent, text=text, command=command)
        self._controls.append(button)
        return button

    def _browse_database(self):
        selected = filedialog.asksaveasfilename(
            title="Open or create SmartRAG database",
            defaultextension=".db",
            filetypes=[("SQLite database", "*.db"), ("All files", "*.*")],
            initialfile=Path(self._db_path.get()).name,
        )
        if selected:
            self._db_path.set(selected)

    def _open_database(self):
        path = self._normalized_db_path()
        if not path:
            return

        def work():
            distiller = self._new_store(path)
            try:
                return {"status": format_store_status(distiller, path),
                        "output": f"\nOpened database: {path}\n"}
            finally:
                self._close_store(distiller)

        self._start_task("Opening database...", work)

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select files to ingest",
            filetypes=[
                ("Supported data",
                 "*.xlsx *.xlsm *.csv *.json *.log *.txt *.dlt *.md *.rst "
                 "*.pdf *.docx *.py *.c *.cpp *.h *.xml *.arxml *.dbc *.db"),
                ("All files", "*.*"),
            ],
        )
        if paths:
            self._ingest(list(paths))

    def _add_folder(self):
        path = filedialog.askdirectory(title="Select a folder to ingest")
        if path:
            self._ingest([path])

    def _ingest(self, paths: list[str]):
        db_path = self._normalized_db_path()
        if not db_path:
            return

        def work():
            distiller = self._new_store(db_path)
            summaries = []
            try:
                for path in paths:
                    stats = distiller.ingest(path, verbose=False)
                    line = (
                        f"{path}\n"
                        f"  {stats['files_ingested']} ingested, "
                        f"{stats.get('files_skipped_unchanged', 0)} unchanged, "
                        f"{stats.get('files_skipped_unsupported', 0)} unsupported, "
                        f"{len(stats.get('errors', []))} errors"
                    )
                    summaries.append(line)
                    summaries.extend(f"    ERROR: {err}"
                                     for err in stats.get("errors", []))
                return {
                    "status": format_store_status(distiller, db_path),
                    "output": "\nIngest complete:\n" + "\n".join(summaries) + "\n",
                }
            finally:
                self._close_store(distiller)

        self._start_task(f"Ingesting {len(paths)} selection(s)...", work)

    def _ask(self):
        question = self._question.get().strip()
        db_path = self._normalized_db_path()
        use_ai = self._use_ai.get()
        if not question or not db_path:
            return

        def work():
            distiller = self._new_store(db_path)
            try:
                result = distiller.answer(question)
                text = (
                    f"\nQUESTION: {question}\n"
                    "GROUNDED ANSWER\n"
                    f"{result.to_text()}\n"
                )
                if use_ai:
                    llm = self._get_llm(distiller)
                    if llm:
                        grounding = result.to_text(show_evidence=True)
                        phrased = distiller._phrase(question, grounding, llm)
                        text += f"\nAI SUMMARY\n{phrased or '(AI returned nothing)'}\n"
                    else:
                        text += "\nAI SUMMARY\nNo AI backend is configured.\n"
                return {"status": format_store_status(distiller, db_path),
                        "output": text}
            finally:
                self._close_store(distiller)

        self._start_task("Retrieving grounded evidence...", work)

    def _search(self):
        entity = self._entity.get().strip() or None
        attribute = self._attribute.get().strip() or None
        db_path = self._normalized_db_path()
        if (not entity and not attribute) or not db_path:
            return

        def work():
            distiller = self._new_store(db_path)
            try:
                hits = distiller.search(entity=entity, attribute=attribute, limit=40)
                lines = [f"\nSEARCH: {len(hits)} fact(s)"]
                lines.extend(
                    f"  {hit.entity} | {hit.attribute}: {hit.value} [{hit.source}]"
                    for hit in hits
                )
                return {"status": format_store_status(distiller, db_path),
                        "output": "\n".join(lines) + "\n"}
            finally:
                self._close_store(distiller)

        self._start_task("Searching facts...", work)

    def _normalized_db_path(self) -> str:
        raw = self._db_path.get().strip()
        if not raw:
            self._status.set("Select a database path first.")
            return ""
        path = Path(raw).expanduser().resolve()
        self._db_path.set(str(path))
        return str(path)

    @staticmethod
    def _new_store(db_path: str) -> SmartRAG:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        return SmartRAG(db_path)

    @staticmethod
    def _close_store(distiller: SmartRAG):
        if distiller.db is not None:
            distiller.db.close()

    def _start_task(self, status: str, work):
        if self._busy:
            self._status.set("Wait for the current operation to finish.")
            return
        self._set_busy(True)
        self._status.set(status)

        def run():
            try:
                self._events.put(("result", work()))
            except Exception as exc:  # noqa: BLE001
                self._events.put(("error", str(exc)))
            finally:
                self._events.put(("done", None))

        threading.Thread(target=run, daemon=True).start()

    def _poll_events(self):
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "result":
                    self._append(payload.get("output", ""))
                    self._status.set(payload.get("status", "Ready."))
                elif kind == "error":
                    self._append(f"\nERROR: {payload}\n")
                    self._status.set("Operation failed; see output.")
                elif kind == "done":
                    self._set_busy(False)
        except queue.Empty:
            pass
        self.after(80, self._poll_events)

    def _set_busy(self, busy: bool):
        self._busy = busy
        state = "disabled" if busy else "normal"
        for control in self._controls:
            control.configure(state=state)

    def _append(self, text: str):
        if not text:
            return
        self._output.configure(state="normal")
        self._output.insert("end", text)
        self._output.see("end")
        self._output.configure(state="disabled")

    def _clear_output(self):
        self._output.configure(state="normal")
        self._output.delete("1.0", "end")
        self._output.configure(state="disabled")

    def _get_llm(self, distiller: SmartRAG):
        direct = getattr(distiller, "llm", None)
        if direct:
            return direct
        if self._llm is not None:
            return self._llm
        try:
            config_path = (Path(os.environ.get("APPDATA", Path.home()))
                           / "RTCAnalyzer" / "config.json")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            base = Path(__file__).resolve().parent.parent
            for path in (str(base), str(base / "src")):
                if path not in sys.path:
                    sys.path.insert(0, path)
            backend = config.get("llm_backend", "")
            if backend == "magica":
                import magica_backend as module
                post = module.make_magica_post_fn(
                    config.get("magica_api_key", ""),
                    config.get("magica_workflow_id", ""),
                    max_wait=200,
                    verbose=False,
                    proxy=config.get("magica_proxy", ""),
                )
            elif backend == "gemini_web":
                import gemini_web_backend as module
                post = module.make_gemini_web_post_fn(
                    proxy=config.get("magica_proxy", ""),
                    verbose=False,
                    model=config.get("gemini_web_model", "") or None,
                )
            else:
                return None

            def call(messages):
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(post(messages))
                finally:
                    loop.close()

            self._llm = call
        except Exception:
            self._llm = None
        return self._llm


DistillGUI = SmartRAGGUI


def main():
    SmartRAGGUI().mainloop()


if __name__ == "__main__":
    main()
