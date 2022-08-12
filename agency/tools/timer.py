import time


class TimeDelta:
    def __init__(self):
        self.reset()

    def reset(self):
        self._start_time = time.time()
        self._last_log_time = time.time()

    def get_delta_in_seconds(self, update: bool = False, from_start: bool = False) -> float:
        dt = 0.0
        if from_start:
            dt = float(time.time() - self._start_time)
        else:
            dt = float(time.time() - self._last_log_time)
        if update:
            self.update()
        return dt

    def update(self):
        self._last_log_time = time.time()

    def get_total_time_as_string(self):
        return time.strftime("%H:%M:%S", time.gmtime(time.time() - self._start_time))
