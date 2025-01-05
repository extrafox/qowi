import sys


def progress_bar(progress, total, bar_width=80):
    filled_length = int(bar_width * progress // total)
    bar = '=' * filled_length + '-' * (bar_width - filled_length)
    sys.stdout.write(f'\r|{bar}| {progress}/{total} ({(progress / total) * 100:.2f}%)')
    sys.stdout.flush()
