import asyncio
from queue import Empty, Queue
import threading
from tkinter import *

gui_queue = Queue()

def updateLabelGui(label_text):
    label.config(text=label_text)

async def updateLabel():
    print("run")
    import random
    while True:
        r = 5*random.random()
        # release control back to event loop while sleeping for r
        await asyncio.sleep(r)
        gui_queue.put(lambda: updateLabelGui(str(r)))
        # release control back to event loop while sleeping for r
        await asyncio.sleep(r)

# http://effbot.org/zone/tkinter-threads.htm
def periodicGuiUpdate():
    while True:
        try:
            fn = gui_queue.get_nowait()
        except Empty:
            break
        fn()
    root.after(100, periodicGuiUpdate)

def doTask():
    asyncio.run_coroutine_threadsafe(updateLabel(), loop)

loop = asyncio.new_event_loop()

# Run the asyncio event loop in a worker thread.
def start_loop():
    asyncio.set_event_loop(loop)
    loop.run_forever()
threading.Thread(target=start_loop).start()

# Run the GUI main loop in the main thread.
root = Tk()
label = Label(root)
label.pack()
button = Button(root, text="Run", command=doTask)
button.pack()
periodicGuiUpdate()
root.mainloop()

# To stop the event loop, call loop.call_soon_threadsafe(loop.stop).
# To start a coroutine from the GUI, call asyncio.run_coroutine_threadsafe.