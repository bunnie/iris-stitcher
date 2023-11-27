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

    def __sub__(self, p):
        return Point(
            self.x - p.x,
            self.y - p.y
        )

    def __neg__(self):
        return Point(0, 0) - self

    def as_tuple(self):
        return (self.x, self.y)

    def as_int_tuple(self):
        return (int(self.x), int(self.y))

    def copy(self):
        return Point(self.x, self.y)

    def __str__(self):
        return f"Point: ({self.x:0.3f}, {self.y:0.3f})"

    def clamp_zero(self):
        if self.x < 0:
            rx = 0
        else:
            rx = self.x
        if self.y < 0:
            ry = 0
        else:
            ry = self.y
        return Point(rx, ry)

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

    def __str__(self):
        return f"Rect: ({self.tl.x:0.3f}, {self.tl.y:0.3f}), ({self.br.x:0.3f}, {self.br.y:0.3f})"

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

    def translate(self, p: Point):
        return Rect(
            self.tl + p,
            self.br + p
        )

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

    def width(self):
        return self.br.x - self.tl.x

    def height(self):
        return self.br.y - self.tl.y

    def center(self):
        return Point(
            (self.br.x + self.tl.x) / 2,
            (self.br.y + self.tl.y) / 2,
        )

    def scale(self, s):
        center = self.center()
        # scale of 0.25: rectangle of width 1 -> rectangle of width 0.25
        return Rect(
            Point(center.x - (self.width() / 2) * s, center.y - (self.height() / 2) * s),
            Point(center.x + (self.width() / 2) * s, center.y + (self.height() / 2) * s)
        )

    def area(self):
        return self.width() * self.height()

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
