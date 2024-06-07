# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 10:48:41 2021

@author: PaulLeroy
"""

import time


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:

    timers = dict()    

    def __init__(self, name=None):
        self._start_time = None
        self.name = name
        
        # Add new named timers to dictionary of timers
        if name:
            self.timers.setdefault(name, 0)

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()


    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        
        if self.name:
            self.timers[self.name] += elapsed_time
        return elapsed_time

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.stop()
