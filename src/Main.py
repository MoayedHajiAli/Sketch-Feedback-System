from Morph import Morph
import pandas as pd
import numpy as np

def main():
    morph = Morph('a.xml', 'b.xml')
    #for obj in morph.original:
    #    obj.visualize()
    morph.start()

if __name__ == '__main__':
    main()