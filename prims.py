# Number of decimal places to round coordinates to prior to
# magnitude comparisons.
ROUNDING = 5

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError

    def __setitem__(self, index, val):
        if index == 0:
            self.x = val
        elif index == 1:
            self.y = val
        else:
            raise IndexError

    def __eq__(self, p):
        return round(self.x, ROUNDING) == round(p.x, ROUNDING) \
            and round(self.y, ROUNDING) == round(p.y, ROUNDING)

    def __add__(self, p):
        return Point(
            self.x + p.x,
            self.y + p.y
        )

    def as_tuple(self):
        return (self.x, self.y)

    def as_int_tuple(self):
        return (int(self.x), int(self.y))

    def copy(self):
        return Point(self.x, self.y)

# All units are in mm
class Rect():
    def __init__(self, tl: Point, br: Point):
        self.tl = tl
        self.br = br
        self.__normalize__()

    def __getitem__(self, index):
        if index == 0:
            return self.tl
        elif index == 1:
            return self.br
        else:
            raise IndexError

    def __setitem__(self, index, val: Point):
        if index == 0:
            self.tl = val
        elif index == 1:
            self.br = val
        else:
            raise IndexError
        self.__normalize__()

    # Used to ensure that coordinates are in tl, br order after updates
    # Coordinate system is:
    # 0, 0 -->
    # | tl
    # v     br
    def __normalize__(self):
        p1 = self.tl.copy()
        p2 = self.br.copy()
        self.tl = Point(min(p1.x, p2.x), min(p1.y, p2.y))
        self.br = Point(max(p1.x, p2.x), max(p1.y, p2.y))

    def __eq__(self, r):
        return r.tl == self.tl and r.br == self.br

    def intersects(self, p: Point):
        return round(self.tl.x, ROUNDING) <= round(p.x, ROUNDING) \
            and round(self.tl.y, ROUNDING) <= round(p.y, ROUNDING) \
            and round(self.br.x, ROUNDING) >= round(p.x, ROUNDING) \
            and round(self.br.y, ROUNDING) >= round(p.y, ROUNDING)

    def intersection(self, r):
        tl = Point(
            max(self.tl.x, r.tl.x),
            max(self.tl.y, r.tl.y)
        )
        br = Point(
            min(self.br.x, r.br.x),
            min(self.br.y, r.br.y)
        )
        if tl.x > br.x or tl.y > br.y:
            return None
        else:
            return Rect(tl, br)

    @staticmethod
    def test():
        r1 = Rect(Point(0, 0), Point(1, 1))

        # Diagonal cases
        r2 = Rect(Point(-1, -1), Point(0.5, 0.5))
        assert r1.intersection(r2) == Rect(Point(0,0), Point(0.5, 0.5))
        r3 = Rect(Point(2, 2), Point(0.5, 0.5))
        assert r1.intersection(r3) == Rect(Point(0.5, 0.5), Point(1, 1))
        r4 = Rect(Point(-1, 2), Point(0.5, 0.5))
        assert r1.intersection(r4) == Rect(Point(0, 1), Point(0.5, 0.5))
        r5 = Rect(Point(2, -1), Point(0.5, 0.5))
        assert r1.intersection(r5) == Rect(Point(0.5, 0.5), Point(1, 0))

        # Vertical straddle cases
        r6 = Rect(Point(-1, -1), Point(0.5, 2))
        assert r1.intersection(r6) == Rect(Point(0, 0), Point(0.5, 1))
        r7 = Rect(Point(0.25, -1), Point(0.5, 2))
        assert r1.intersection(r7) == Rect(Point(0.25, 0), Point(0.5, 1))
        r8 = Rect(Point(0.5, -1), Point(2, 2))
        assert r1.intersection(r8) == Rect(Point(0.5, 0), Point(1, 1))

        # Horizontal straddle cases
        r9 = Rect(Point(-1, 2), Point(2, 0.5))
        assert r1.intersection(r9) == Rect(Point(0, 0.5), Point(1, 1))
        r10 = Rect(Point(-1, 0.75), Point(2, 0.25))
        assert r1.intersection(r10) == Rect(Point(0, 0.25), Point(1, 0.75))
        r11 = Rect(Point(-1, 0.5), Point(2, -1))
        assert r1.intersection(r11) == Rect(Point(0, 0.5), Point(1, 0))

        # Inside case
        r12 = Rect(Point(0.25, 0.25), Point(0.5, 0.5))
        assert r1.intersection(r12) == Rect(Point(0.25, 0.25), Point(0.5, 0.5))

        # Outside case
        r13 = Rect(Point(-1, -1), Point(2, 2))
        assert r1.intersection(r13) == Rect(Point(0, 0), Point(1, 1))

        # No intersection case
        r14 = Rect(Point(3, 3), Point(4, 4))
        assert r1.intersection(r14) == None

        # Identity case
        r15 = Rect(Point(0, 0), Point(1, 1))
        assert r1.intersection(r15) == Rect(Point(0, 0), Point(1, 1))
