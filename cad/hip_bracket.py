import math

from build123d import Box, Cone, Cylinder, Plane, Polyline, Pos, Rot, Torus, extrude, make_face

AXIS = (0.0, -91.0)
BUILD_DIR = "+x"

T = 10.0          # plate thickness (x 0..T); the actuator seat is the x=T face
RD = 44.1         # seat disc radius
PAD_W, PAD_H = 54.0, 14.0   # belt pad
WEB_W = 44.0      # web width
R_DISC = 12.0     # in-plane fillet, web to disc
REC_D, REC_TOP, REC_RAMP = 6.8, 0.0, 30.0   # back-face recess of the lower disc: depth, start above axis, ramp length
PAD_T, PAD_Z0, PAD_Z1 = 3.2, -27.0, -17.0   # front face stepped down to PAD_T above PAD_Z1, ramp from full T at PAD_Z0
EXT_STOP, FLEX_STOP = -34.95, 139.0   # hard stops (deg): round pegs the link bar runs into
PEG_R, PEG_FILLET, PEG_EXT_AT, PEG_FLEX_AT = 3.0, 2.0, 30.0, 34.0   # peg radius, root fillet, radial positions
DBL, DBL_BOT = 2.8, 22.0   # back-face doubler on the web and neck: extra thickness toward the body (0 = none), lower end above the axis
SIDE_Y0 = 30.0   # back-face side chamfers of the disc start at |y| = SIDE_Y0 (0 = none)
LAND_R, LAND_FLARE = 7.0, 0.8   # full-thickness land under the extension peg; radius growth per mm of depth (0 = cylinder)


def xcyl(r, x0, x1, y=0.0, z=0.0):
    return Pos((x0 + x1) / 2, y, z) * Rot(0, 90, 0) * Cylinder(r, x1 - x0)


def yz_prism(pts, x0, x1):
    face = make_face(Polyline(*[Plane.YZ.from_local_coords(p) for p in pts], close=True))
    return Pos(x0, 0, 0) * extrude(face, x1 - x0, dir=(1, 0, 0))


def xz_prism(pts, half_y):
    face = make_face(Polyline(*[Plane.XZ.from_local_coords(p) for p in pts], close=True))
    return extrude(face, half_y, both=True)


def arc(c, r, a0, a1, n):
    return [(c[0] + r * math.cos(math.radians(a0 + (a1 - a0) * k / n)),
             c[1] + r * math.sin(math.radians(a0 + (a1 - a0) * k / n))) for k in range(n + 1)]


def web_outline(ya, za):
    """Web from the pad to the disc with a tangent fillet into the disc (right half, mirrored)."""
    hw = WEB_W / 2
    zc = za + math.sqrt((RD + R_DISC) ** 2 - (hw + R_DISC) ** 2)
    c1 = (hw + R_DISC, zc)
    a_t = math.degrees(math.atan2(za - zc, -(hw + R_DISC))) % 360
    right = [(hw, 0.0)] + arc(c1, R_DISC, 180, a_t + 3, 10) + [(0.0, za)]
    left = [(-y, z) for y, z in reversed(right[:-1])]
    return [(y + ya, z) for y, z in right + left]


def xy_prism(pts, z0, z1):
    face = make_face(Polyline(*[Plane.XY.from_local_coords(p) for p in pts], close=True))
    return Pos(0, 0, z0) * extrude(face, z1 - z0, dir=(0, 0, 1))


def peg_centre(theta_stop, r_c, ya, za):
    """Centre of a round peg at radius r_c that the bar edge (13 mm off the link centreline) touches at theta_stop."""
    sgn = 1.0 if theta_stop > 0 else -1.0
    phi = math.radians(theta_stop + sgn * math.degrees(math.asin((13.0 + PEG_R) / r_c)))
    return ya + r_c * math.sin(phi), za - r_c * math.cos(phi)


def peg(y, z, x_root):
    """Round stop peg from x=-13 to the plate face at x_root, with a concave root fillet."""
    ro = PEG_R + PEG_FILLET
    flare = xcyl(ro, x_root - PEG_FILLET, x_root + 1, y, z) - Pos(x_root - PEG_FILLET, y, z) * Rot(0, 90, 0) * Torus(ro, PEG_FILLET)
    return xcyl(PEG_R, -13, x_root + 1, y, z) + flare


def build():
    ya, za = AXIS
    part = Pos(T / 2, 0, 0) * Box(T, PAD_W, PAD_H)
    part += yz_prism(web_outline(ya, za), 0, T)
    part += xcyl(RD, 0, T, ya, za)
    if REC_D > 0:
        zt = za + REC_TOP
        part -= xz_prism([(0, zt), (REC_D, zt - REC_RAMP), (REC_D, za - RD - 2), (-1, za - RD - 2), (-1, zt)], 60)
    if PAD_T < T:
        top = PAD_H / 2 + 1
        part -= xz_prism([(PAD_T, top), (PAD_T, PAD_Z1), (T, PAD_Z0), (T + 1, PAD_Z0), (T + 1, top)], 40)
    if SIDE_Y0 > 0:
        for sg in (1.0, -1.0):
            part -= xy_prism([(0, sg * SIDE_Y0), (REC_D, sg * RD), (REC_D, sg * 60), (-1, sg * 60), (-1, sg * SIDE_Y0)], za - 50, za + 50)
    if DBL > 0:
        zt, zb, yo = -31.0, za + DBL_BOT, WEB_W / 2 - 0.5
        yi = yo - 2 * DBL
        part += (xz_prism([(0.5, zt), (0, zt), (-DBL, zt - 2 * DBL), (-DBL, zb + 2 * DBL), (0, zb), (0.5, zb)], yo)
                 & xy_prism([(0.5, -yo), (0, -yo), (-DBL, -yi), (-DBL, yi), (0, yo), (0.5, yo)], zb, zt))
    part += peg(*peg_centre(FLEX_STOP, PEG_FLEX_AT, ya, za), -DBL)
    ye, ze = peg_centre(EXT_STOP, PEG_EXT_AT, ya, za)
    if LAND_R > 0 and LAND_FLARE > 0:
        part += Pos((T - 1) / 2, ye, ze) * Rot(0, 90, 0) * Cone(LAND_R, LAND_R + LAND_FLARE * (T - 1), T - 1)
    elif LAND_R > 0:
        part += xcyl(LAND_R, 0, T - 1, ye, ze)
    part += peg(ye, ze, 0.0 if LAND_R > 0 else REC_D)
    part -= xcyl(21, -14, T + 1, ya, za)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        part -= xcyl(2.25, -14, T + 1, ya + 35 * math.sin(a), za - 35 * math.cos(a))
    for y in (-20.0, 20.0):
        part -= xcyl(2.75, -1, 26, y, 0.0)
    return part
