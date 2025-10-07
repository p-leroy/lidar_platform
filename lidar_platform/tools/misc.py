# coding: utf-8
# Paul Leroy

import datetime
import glob
import os
import re
import subprocess
import sys
import time
import shlex


def exists(path):
    if not os.path.exists(path):
        print(f'file does not exists! {path}')
        return False
    else:
        return True


def delete_file(files):
    for file in files:
        try:
            os.remove(file)
        except:
            pass


def list_files(idir, patterns):
    list_ = []

    if type(patterns) is list:
        for pattern in patterns:
            to_append = glob.glob(os.path.join(idir, pattern))
            if len(to_append) != 0:
                list_.extend(to_append)
    else:
        list_.extend(glob.glob(os.path.join(idir, patterns)))

    return list_


def head_tail_root_ext(path):
    head, tail = os.path.split(path)
    root, ext = os.path.splitext(tail)
    return head, tail, root, ext


def to_bool(str_):
    if 'false' in str_:
        str_ = str_.replace('false', 'False')
    elif 'true' in str_:
        str_ = str_.replace('true', 'True')
    return eval(str_)


def to_str(bool_):
    if bool_ is True:
        ret = 'true'
    else:
        ret = 'false'
    return ret


def pyuic5(ui, debug=False):
    head, tail = os.path.split(ui)
    root, ext = os.path.splitext(tail)
    out = os.path.join(head, 'Ui_' + root + '.py')
    cmd = [sys.executable(), '-m', 'PyQt5.uic.pyuic', ui, '-o', out]
    print(cmd)
    run(cmd, debug=debug)


def run(cmd, shell=False, advanced=True, verbose=True):
    # The recommended approach to invoking subprocesses is to use the run() function for all use cases it can handle.
    # For more advanced use cases, the underlying Popen interface can be used directly.
    # The only time you need to specify shell=True on Windows is when the command you wish to execute is built into
    # the shell (e.g. dir or copy)

    if advanced:
        if verbose:
            print('[CMD] ' + shlex.join(cmd))
        # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, shell=shell)
        # while process.poll() is None:
        #     for line in process.stdout.readlines():
        #         if verbose is True:
        #             print(line.strip())
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        while process.poll() is None:
            # Use read1() instead of read() or Popen.communicate() as both blocks until EOF
            # https://docs.python.org/3/library/io.html#io.BufferedIOBase.read1
            text = process.stdout.read1().decode("utf-8")
            if verbose is True:
                print(text, end='', flush=True)
        # Process has finished, read the rest of the output
        for line in process.stdout.readlines():
            if verbose is True:
                print(line.strip())
            else:
                pass
        process.wait()
        ret = process.returncode

    else:
        print("Run command in a subprocess:")
        print(cmd)
        completed_process = subprocess.run(cmd, capture_output=True, shell=shell)
        ret = completed_process.returncode

    return ret


def run_alt(cmd):
    print("Run command in a subprocess:")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    while process.poll() is None:
        # Use read1() instead of read() or Popen.communicate() as both blocks until EOF
        # https://docs.python.org/3/library/io.html#io.BufferedIOBase.read1
        text = process.stdout.read1().decode("utf-8")
        print(text, end='', flush=True)


def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def snake_to_camel(name):
    return ''.join(word.title() for word in name.split('_'))


class Timing(object):
    def __init__(self, length, step=20):
        self.length = length
        if step > 5:
            listVerbose = [2] + list(range(step, 100, step)) + [98]
        else:
            listVerbose = list(range(step, 100, step)) + [98]
        self.percent = [int(i * self.length / 100) for i in listVerbose]
        self.start = time.time()

    def timer(self, idx):
        if idx in self.percent:
            duration = round(time.time() - self.start, 1)
            remain = round(duration*(self.length / idx - 1), 1)
            msg = str(idx) + " in " + str(duration)+"sec - remaining " + str(remain) + "sec"
            return msg


class DATE(object):
    def __init__(self):
        today = datetime.datetime.now().timetuple()
        self.year = today.tm_year
        self.day = today.tm_mday
        self.month = today.tm_mon
        self.date = str(str(self.year)
                        + "-"
                        + str("0" * (2 - len(str(self.month)))
                        + str(self.month))
                        + "-" + str("0" * (2 - len(str(self.day)))
                        + str(self.day)))
        self.time = str("0" * (2 - len(str(today.tm_hour))) + str(today.tm_hour)) \
                    + "h" + str("0" * (2 - len(str(today.tm_min))) + str(today.tm_min))
