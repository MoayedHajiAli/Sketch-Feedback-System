import sys
sys.path.insert(0, '../')

from sketch_object.UnlabeledObject import UnlabeledObject
from sketch_object.Point import Point
from sketch_object.Stroke import Stroke
import unittest

class UnlabeledObjectTest(unittest.TestCase):
    def test_eq(self):
        obj1 = UnlabeledObject([Stroke([Point(1, 2, 3), Point(0, 0, 1)])])
        obj2 = UnlabeledObject([Stroke([Point(1, 2, 3), Point(0, 0, 1)])])
        assert obj1 == obj2

        # points will be reordered according to time
        obj1 = UnlabeledObject([Stroke([Point(1, 2, 2), Point(0, 0, 1)])])
        obj2 = UnlabeledObject([Stroke([Point(0, 0, 1), Point(1, 2, 3)])])
        assert obj1 == obj2

        obj1 = UnlabeledObject([Stroke([Point(1, 2, 0), Point(0, 0, 1)])])
        obj2 = UnlabeledObject([Stroke([Point(0, 0, 1), Point(1, 2, 3)])])
        assert obj1 != obj2

if __name__ == '__main__':
    unittest.main()