
import json
import queue
import threading
import webbrowser
from pathlib import Path

from pipeline import run, DEFAULTS, validate_config


# Convert form entries into validated pipeline settings
def options(lead, review, horizon, products, stock_text):
    config_path = Path(__file__).resolve().parent / "config.json"
    base = json.loads(config_path.read_text()) if config_path.exists() else {}
    config = {
        **DEFAULTS,
        **base,
        "lead_days": int(lead),
        "review_days": int(review),
        "future_days": int(horizon),
        "products": [x.strip().upper() for x in products.split(",") if x.strip()],
    }
    config["stock_overrides"] = json.loads(stock_text or "{}")
    validate_config(config)
    return config



# Objective 9: Build and run the desktop interface
def main():
    # Analytical code and tests can import options() without installing Tk
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    except ImportError as exc:
        raise RuntimeError(
            "The desktop interface requires Tkinter. Install Python with Tk support, "
            "or run pipeline.py from the command line."
        ) from exc

    # Create the window and shared layout
    root = tk.Tk()
    root.title("Retail forecasting and inventory study")
    root.geometry("880x650")
    root.minsize(760, 550)
    style = ttk.Style()
    style.theme_use("clam")
    frame = ttk.Frame(root, padding=22)
    frame.pack(fill="both", expand=True)
    ttk.Label(
        frame, text="Retail forecasting & inventory", font=("Arial", 21, "bold")
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
    file = tk.StringVar()
    out = tk.StringVar(value=str(Path(__file__).resolve().parent / "output"))

    # Reusable input row with an optional browse button
    def field(row, label, var, button=None):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=var, width=65).grid(
            row=row, column=1, sticky="ew", pady=5
        )
        if button:
            ttk.Button(frame, text="Browse", command=button).grid(
                row=row, column=2, padx=7
            )

    field(
        2,
        "Input workbook / CSV",
        file,
        lambda: file.set(
            filedialog.askopenfilename(filetypes=[("Retail data", "*.xlsx *.csv")])
            or file.get()
        ),
    )
    field(
        3,
        "Results folder",
        out,
        lambda: out.set(filedialog.askdirectory() or out.get()),
    )
    lead = tk.StringVar(value="7")
    review = tk.StringVar(value="7")
    horizon = tk.StringVar(value="14")
    products = tk.StringVar()
    stock = tk.StringVar(value="{}")
    field(4, "Lead time (calendar days)", lead)
    field(5, "Review interval (days)", review)
    field(6, "Future forecast days", horizon)
    field(7, "Product codes (optional)", products)
    field(8, "Current stock JSON", stock)
    ttk.Label(
        frame,
        text='Leave product codes blank for nine products selected from training data. Example stock: {"85123A": 100}.\nInventory costs and missing stock are illustrative assumptions; detailed settings are in config.json.',
        wraplength=800,
    ).grid(row=9, column=0, columnspan=3, sticky="w", pady=10)
    log = tk.Text(frame, height=12, wrap="word", font=("Consolas", 10))
    log.grid(row=11, column=0, columnspan=3, sticky="nsew", pady=10)

    # Pass worker messages to the main GUI thread safely
    messages = queue.Queue()
    last_output = [None]

    # Validate inputs before starting the background calculation
    def start():
        try:
            if not Path(file.get()).is_file():
                raise ValueError("Choose an existing input workbook or CSV.")
            config = options(
                lead.get(), review.get(), horizon.get(), products.get(), stock.get()
            )
            input_path = file.get()
            output = out.get()
            if not output:
                raise ValueError("Choose a results folder.")
        except Exception as e:
            messagebox.showerror("Check inputs", str(e))
            return
        run_button.config(state="disabled")
        log.delete("1.0", "end")

        def worker():
            try:
                run(
                    input_path,
                    output,
                    config,
                    progress=lambda s: messages.put(("log", s)),
                )
                from local_excel import export_excel

                export_excel(output)
                messages.put(("done", output))
            except Exception as e:
                messages.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    # Handle progress and completion on the GUI thread
    def poll():
        while not messages.empty():
            kind, msg = messages.get()
            log.insert("end", msg + "\n")
            log.see("end")
            if kind in ("done", "error"):
                run_button.config(state="normal")
                if kind == "done":
                    last_output[0] = Path(msg)
                else:
                    messagebox.showerror("Run failed", msg)
        root.after(150, poll)

    # Open the report from the last successful run
    def open_report():
        path = (last_output[0] or Path(out.get())) / "report.html"
        if path.exists():
            webbrowser.open(path.resolve().as_uri())
        else:
            messagebox.showinfo("Report", "Run the pipeline first.")

    # Wire the buttons and enter the GUI event loop
    buttons = ttk.Frame(frame)
    buttons.grid(row=10, column=0, columnspan=3, sticky="w")
    run_button = ttk.Button(
        buttons, text="Run analysis and generate report", command=start
    )
    run_button.pack(side="left", padx=(0, 10))
    ttk.Button(buttons, text="Open report", command=open_report).pack(side="left")
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(11, weight=1)
    poll()
    root.mainloop()


if __name__ == "__main__":
    main()
