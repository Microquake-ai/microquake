import time
import sys


class ProgressBar(object):
    """
    This object display a simple command line progress bar to wait in style.
    """
    def __init__(self, max = 100, size = 50, char = "#", caption = "Progress"):
        self.max = max
        self.value = 0
        self.caption = caption
        self.size = size
        self.start_time = time.time()
        self.char = char

        if self.max == 0:
            self.draw = self.__emptydraw__

        self.draw()

    def draw(self):

        now = time.time()
        if self.value != 0:
            remaining = ((now - self.start_time) / self.value) * (self.max - self.value)
        else:
            remaining = 0
        pos = self.size * self.value / self.max
        eta = "%d of %d %3.1fs (%2.0f%%)" % (self.value, self.max, remaining, 100 * self.value / self.max)
        progress = self.char * pos + (self.size - pos) * " "
        progress_string = "[%s]" % (progress)
        eta_string = "ETA %s" % (eta)
        caption_string = " " +  self.caption
        sys.stderr.write("%s : %s %s\r" % (caption_string, progress_string, eta_string))
        if self.value >= self.max:
            sys.stderr.write("\n -- TOTAL TIME  : %2.4fs -- \n" % (now - self.start_time))

    def __emptydraw__(self):
        pass

    def __call__(self, update = 1):
        self.update(update = update)

    def update(self, update = 1):
        uvalue = self.value + update
        self.value = min(uvalue, self.max)
        self.draw()

    def set_value(self, value):
        self.value = min(value, sefl.max)
        self.draw()


class Timer(object):
    """
    This class is a simple timer for the sake of simplicity. This also provides
    simple statistics. This work with the python 'with statement'.
    """
    total = 0
    t = None
    n = 0

    def start(self):
        """
        Record the current time
        """
        if self.t is None:
            self.t = time.time()
        else:
            raise RuntimeError("Timer already started")


    def stop(self, *args):
        """
        Stop the timer and record the execution
        """
        if self.t is not None:
            self.total += time.time() - self.t
            self.n += 1
            self.t = None
        else:
            raise RuntimeError("Timer not started")

    __enter__ = start
    __exit__ = stop

    def mean(self):
        """
        Return the average runtime of this timer
        """
        return self.total / self.n

    def reset(self):
        """
        Reset the statistics
        """
        self.n = self.total = 0


if __name__ == "__main__":
    a = ProgressBar()
    for i in range(100):
        time.sleep(0.02)
        a()
