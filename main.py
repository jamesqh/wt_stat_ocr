import asyncio
import base64
from collections import defaultdict
from io import BytesIO
import json
import os
import pandas as pd
from PIL import Image as PILImage, ImageTk
from queue import Empty, Queue
from slugify import slugify
import threading
import time
from tkinter import *
from tkinter.ttk import *

from ocr_interface import ScreenshotOCRInterface


class NavButtons(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.read_screenshots_button = Button(
            self, text="Read screenshots", command=self.click_read_screenshots_button
        )
        self.inspect_data_button = Button(
            self, text="Inspect data", command=self.click_inspect_data_button
        )
        self.export_stats_button = Button(
            self, text="Export stats", command=self.click_export_stats_button
        )
        self.read_screenshots_button.pack(side="left")
        self.inspect_data_button.pack(side="left")
        self.export_stats_button.pack(side="left")

    def click_read_screenshots_button(self):
        self.read_screenshots_button.config(state="disabled")
        self.inspect_data_button.config(state="normal")
        self.export_stats_button.config(state="normal")
        self.parent.populate_main_panel(ReadScreenshots)

    def click_inspect_data_button(self):
        self.read_screenshots_button.config(state="normal")
        self.inspect_data_button.config(state="disabled")
        self.export_stats_button.config(state="normal")
        self.parent.populate_main_panel(InspectData)

    def click_export_stats_button(self):
        self.read_screenshots_button.config(state="normal")
        self.inspect_data_button.config(state="normal")
        self.export_stats_button.config(state="disabled")


class ReadScreenshots(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.queue = Queue()
        self.curr_screenshot_index = 0
        self.screenshot_batch_label = Label(self, text="No screenshot processing")
        # self.screenshot_progress_bar = Progressbar(self.parent)
        self.current_screenshot_label = Label(self)
        self.reference_img_preview_label = Label(self)
        self.data_preview_label = Label(self)
        self.screenshot_batch_label.pack()
        # self.screenshot_progress_bar.pack()
        self.current_screenshot_label.pack()
        self.reference_img_preview_label.pack()
        self.data_preview_label.pack()
        Button(self, text="Go", command=self.start_screenshot_read).pack()

    def start_screenshot_read(self):
        reader = ScreenshotReader(self.queue, None)

        async def stupid(coro, *args, **kwargs):
            task = loop.create_task(coro(*args, **kwargs))
            return task

        asyncio.run_coroutine_threadsafe(
            stupid(reader.read_screenshots, "screenshots"), loop
        )
        self.periodic_gui_update()

    def periodic_gui_update(self):
        handlers = {
            "update_screenshot_batch_progress": self.update_screenshot_batch_progress,
            "update_curr_screenshot_progress": self.update_curr_screenshot_progress,
            "update_data_preview": self.update_data_preview,
            "update_row_reference_img": self.update_row_reference_img,
            "clear_data_preview": self.clear_data_preview,
        }
        while True:
            try:
                msg = self.queue.get_nowait()
            except Empty:
                break
            f = handlers[msg["tag"]]
            f(msg["args"])
        self.after(100, self.periodic_gui_update)

    def update_screenshot_batch_progress(self, args):
        curr, total = args
        self.screenshot_batch_label.config(
            text=f"Reading {curr} of {total} screenshots"
        )

    def update_curr_screenshot_progress(self, args):
        curr, total = args
        self.current_screenshot_label.config(
            text=f"Interpreting {curr} of {total} rows in current screenshot"
        )

    def update_data_preview(self, args):
        record = args
        text = []
        for label in [
            "row_num",
            "rank",
            "country",
            "name",
            "wins",
            "battles",
            "win_percent",
            "respawns",
            "deaths",
            "air_kills",
            "ground_kills",
            "boat_kills",
            "sl_earned",
            "rp_earned",
        ]:
            if label in record:
                text.append("{0}: {1}".format(label, record[label]))
        self.data_preview_label.config(text=", ".join(text))

    def clear_data_preview(self, args):
        self.data_preview_label.config(text="")

    def update_row_reference_img(self, args):
        img_enc_b64 = args
        img_enc = base64.b64decode(img_enc_b64)
        img = PILImage.open(BytesIO(img_enc))
        imgtk = ImageTk.PhotoImage(image=img)
        self.reference_img_preview_label.config(image=imgtk)
        self.reference_img_preview_label.image = imgtk


class ScreenshotReader:
    def __init__(self, queue, file_handler):
        self.queue = queue
        if file_handler is None:
            file_handler = lambda: open("raw_stats.json", "a", encoding="utf-8")
        self.file_handler = file_handler

    async def read_screenshots(self, screenshot_dir_path):
        screenshots = os.listdir(screenshot_dir_path)
        num_screenshots = len(screenshots)
        for i, screenshot_path in enumerate(screenshots):
            self.queue.put(
                {
                    "tag": "update_screenshot_batch_progress",
                    "args": (i + 1, num_screenshots),
                }
            )
            await self.read_screenshot(
                os.path.join(screenshot_dir_path, screenshot_path)
            )

    async def read_screenshot(self, screenshot_path):
        screenshot_reader = ScreenshotOCRInterface(
            screenshot_path, queue=self.queue, debug=True
        )
        screenshot_reader.find_rows()
        num_rows = len(screenshot_reader.rows)
        for i, row in enumerate(screenshot_reader.rows):
            self.queue.put(
                {"tag": "update_curr_screenshot_progress", "args": (i + 1, num_rows)}
            )
            record = screenshot_reader.ocr_row(row)
            with self.file_handler() as raw_stat_file:
                raw_stat_file.write(json.dumps(record) + "\n")
                # df = pd.DataFrame.from_dict(record)
                # Write header only if f.tell()==0 (file did not exist)
                # df.to_csv(raw_stat_file, index=False, header=raw_stat_file.tell() == 0)
            # Release control back to the event loop between rows so it can handle shutdown if GUI closed
            await asyncio.sleep(0.001)


class CorrectNames(Frame):
    def __init__(self, parent, records, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        self.pack()
        with open("vehicles.csv", encoding="utf-8") as f:
            self.vehicles = pd.read_csv(f)
        self.records = records
        self.next_record = 0
        self.aliases = defaultdict(lambda: [])
        try:
            with open("aliases.json") as f:
                for k, v in json.load(f).items():
                    for alias in v:
                        self.aliases[alias].append(k)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            pass
        self.seen_names = set()
        self.next_vehicle()

    def next_vehicle(self):
        self.next_record += 1
        if self.next_record <= len(self.records):
            self.handle_record(self.records[self.next_record - 1])
        else:
            print("Done!")

    def handle_record(self, record):
        for child in self.winfo_children():
            child.destroy()
        name_guesses = record["name"]
        rank = int(record["rank"])
        country = record["country"]
        possible_names = defaultdict(lambda: [])
        df = self.vehicles.loc[
            (self.vehicles["rank"] == rank)
            & (self.vehicles["country"] == country)
            & (~self.vehicles["short_name"].isin(self.seen_names))
        ]
        for v_type, s_name in zip(df["type"], df["short_name"]):
            possible_names[v_type].append(s_name)
        for name_guess in name_guesses:
            if any(name_guess in v for v in possible_names.values()):
                self.seen_names.add(name_guess)
                self.next_vehicle()
                return
            if name_guess in self.aliases.keys():
                if len(self.aliases[name_guess]) == 1:
                    record["name"] = self.aliases[name_guess][
                        0
                    ]
                    self.next_vehicle()
                    return
                else:
                    # TODO: actually handle this
                    print(
                        "Multiple valid aliases, oh no!",
                        name_guess,
                        self.aliases[name_guess],
                    )
        img_enc_b64 = record["b64_img_preview"]
        img_enc = base64.b64decode(img_enc_b64)
        img = PILImage.open(BytesIO(img_enc))
        # img.save(f"{slugify(name_guesses[0])}.jpg")
        imgtk = ImageTk.PhotoImage(image=img)
        Label(self, text="Unable to positively identify this vehicle").pack()
        photo_label = Label(self, image=imgtk)
        photo_label.image = imgtk
        photo_label.pack()
        Label(self, text=f"I read it as {name_guesses}").pack()
        Label(self, text="Please select an option from below").pack()
        Frame(self).pack(side=LEFT, fill=BOTH, expand=1)
        for column in possible_names.keys():

            def callback(name):
                def inner_function():
                    record["name"] = name
                    self.seen_names.add(name)
                    self.aliases[name].append(name_guess)
                    with open("aliases.json", "w") as f:
                        json.dump(self.aliases, f)
                    print(self.winfo_children())
                    for child in self.winfo_children():
                        child.destroy()
                    print(self.winfo_children())
                    self.next_vehicle()
                    return

                return inner_function

            frame = Frame(self)
            Label(frame, text=column).pack()
            for name in possible_names[column]:
                Button(frame, text=name, command=callback(name)).pack(fill="x")
            frame.pack(side=LEFT, anchor=NE)
        Frame(self).pack(side=LEFT, fill=BOTH, expand=1)


class InspectData(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        # self.scrollbar = Scrollbar(self)
        # self.data_treeview = Treeview(self, yscrollcommand=self.scrollbar.set)
        # self.scrollbar.config(command=self.data_treeview.yview)
        self.stats = None
        # for record in self.data:
        #     self.data_treeview.insert("", "end", text=str(record["name"]))
        # self.data_treeview.pack(side="left")
        # self.scrollbar.pack(side="left", fill="y")
        Button(self, text="Load data", command=self.load_data).pack()

    def load_data(self):
        for child in self.winfo_children():
            child.destroy()
        records = []
        with open("raw_stats.json", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        correct_names = CorrectNames(self, records)


class MainApplication(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.nav_buttons = NavButtons(self)
        self.nav_buttons.pack()
        self.main_panel = Frame(self)
        self.main_panel.pack(side="bottom")

    def populate_main_panel(self, component):
        for widget in self.main_panel.winfo_children():
            widget.destroy()
        widget = component(self.main_panel)
        widget.pack()


if __name__ == "__main__":
    print("I am running")

    async def shutdown(loop):
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()

    def handle_exception(loop, context):
        loop.default_exception_handler(context)
        msg = context.get("exception", context["message"])
        print("Custom exception handler")
        print(msg)
        asyncio.create_task(shutdown(loop))
        end_loop()
        import sys

        sys.exit()

    loop = asyncio.new_event_loop()
    loop.set_exception_handler(handle_exception)
    loop.set_debug(True)

    def start_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=start_loop)
    thread.start()

    root = Tk()

    def end_loop():
        asyncio.set_event_loop(loop)
        root.destroy()
        loop.call_soon_threadsafe(loop.stop)

    root.protocol("WM_DELETE_WINDOW", end_loop)
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
