import os

class Logger:
    def __init__(self, exp, width=100):
        self.width = width
        self.tqdm_active = False

        self.dir = os.path.join('runs', f"{exp}")
        os.makedirs(self.dir, exist_ok=True)

        if os.path.exists(os.path.join(self.dir, "log.txt")):
            os.remove(os.path.join(self.dir, "log.txt"))


    def __call__(self, message):
        self.bar_active = False

        with open(os.path.join(self.dir, "log.txt"), "a") as f:
            f.write(message + "\n")
            print(message)


    def tqdm(self, progress, total):
        with open(os.path.join(self.dir, "log.txt"), "a") as f:
            if not self.tqdm_active:
                self.tqdm_active = True
            else:
                self._carriage_return()
                print("\r", end="")

            f.write(f"|{'=' * int(self.width * progress / total)}{' ' * (self.width - int(self.width * progress / total))}|")
            print(f"|{'=' * int(self.width * progress / total)}{' ' * (self.width - int(self.width * progress / total))}|", end="")

            if progress == total:
                self.tqdm_active = False
                f.write("\n")
                print("\n", end="")


    def _carriage_return(self):
        with open(os.path.join(self.dir, "log.txt"), 'rb+') as f:
            f.seek(0, 2)
            pos = f.tell() - 1

            while pos > 0:
                f.seek(pos)
                if f.read(1) == b'\n':
                    break
                pos -= 1

            f.truncate(pos + 1 if pos > 0 else 0)
