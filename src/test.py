from progress.bar import Bar
import time


bar = Bar("title", max = 20)
for _ in range(20):
    bar.next()
    time.sleep(0.1)
bar.finish()
