from .component import *

import sys
import os
import queue
from typing import Type
from collections import defaultdict
import threading
from dataclasses import is_dataclass
from enum import Enum
import traceback

class Bus:
    def __init__(self):
        self.subscribers: dict[Type,list[Sink]] = defaultdict(list)
        self.queue = queue.Queue()
        self._running = False
        self._thread = None

    def subscribe(self, data_type: Type, sink: Sink):
        if is_dataclass(data_type):
            self.subscribers[data_type.__name__].append(sink)
        if isinstance(data_type,Enum):
            self.subscribers[data_type].append(sink)

    def publish(self,data):
        raise NotImplementedError

    def _worker(self):
        raise NotImplementedError

    def start(self):
        self._running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.queue.put(None)
        if self.thread:
            self.thread.join()

class DataBus(Bus):
    def __init__(self):
        super().__init__()

    def print_info(self):
        print(f"Databus is listening to:\n {self.subscribers.keys()} \n")
        for message_type, subs in self.subscribers.items():
            print(f"\nfor message type {message_type}\n" + "=" * 40)
            for i, sub in enumerate(subs):
                print(f"({i + 1}) Subscriber : {sub.__class__.__name__}")
    
    def publish(self, data):
        self.queue.put(data)
    
    def _worker(self):
        while self._running:
            item = self.queue.get()

            if item is None:
                break

            try:
                
                if is_dataclass(item):
                    for sink in self.subscribers.get(item.__class__.__name__,[]):
                        sink.process(item)

                elif isinstance(item,Enum):
                    for sink in self.subscribers.get(item,[]):
                        sink.process(item)

                else:
                    sys.stderr.write(f"[DataBus] Warning: Unhandled data type published: {type(item)}\n")

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

                sys.stderr.write(f"[DataBus] Error in subscriber {sink.__class__.__name__}: {e}\n")
                sys.stderr.write(f"  File: {fname}, Line: {exc_tb.tb_lineno}\n")
                traceback.print_exc()

            finally:
                self.queue.task_done()