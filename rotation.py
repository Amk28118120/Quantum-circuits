import numpy as np

# Use a single complex dtype for numpy everywhere.
DTYPE = np.complex128

INV_SQRT2 = 1.0 / np.sqrt(2.0)
H = INV_SQRT2 * np.array([[1, 1], [1, -1]], dtype=DTYPE)

# LAMBDA_PI is the base rotation angle realized by the H/T building blocks:
# cos(LAMBDA_PI) = cos^2(pi/8) = (1 + 1/sqrt2)/2. Because LAMBDA_PI / (2 pi) is
# irrational, the multiples {k * LAMBDA_PI mod 2 pi} densely fill [0, 2 pi).
LAMBDA_PI = np.arccos((1.0 + INV_SQRT2) / 2.0)
TWO_PI = 2.0 * np.pi


class Bloch:
    """Axis-angle (Bloch) form of a 2x2 unitary G:

        G = e^{i alpha} (cos(theta/2) I - i sin(theta/2) (n . sigma))

    i.e. a global phase e^{i alpha} times a rotation by angle `theta` about the
    Bloch-sphere axis `n`. Here (n . sigma) = n_x X + n_y Y + n_z Z.
    """

    def __init__(self):
        self.alpha = 0.0
        self.n = None
        self.theta = 0.0
k = 1 / np.tan(np.pi/8)

n1 = np.array([
    -k,
     1,
     k
], dtype=float)

n2 = np.array([
     1/np.sqrt(2),
     np.sqrt(2)*k,
    -1/np.sqrt(2)
], dtype=float)

n1 /= np.linalg.norm(n1)
n2 /= np.linalg.norm(n2)
a1 = -n1
a2 = -n2
a3 = np.cross(a1, a2)
a3 /= np.linalg.norm(a3)

def to_bloch(g: np.ndarray) -> Bloch:
    # ----- Step 1: Global phase -----
    alpha = 0.5 * np.angle(np.linalg.det(g))

    # ----- Step 2: Remove global phase -----
    u = np.exp(-1j * alpha) * g

    # ----- Step 3: Rotation angle -----
    c = np.real(np.trace(u) / 2.0)

    # Numerical safety
    c = np.clip(c, -1.0, 1.0)

    theta = 2.0 * np.arccos(c)
    s = np.sin(theta / 2)

    # ----- Step 4: Rotation axis -----
    if np.isclose(s, 0):
        n = np.array([1.0, 0.0, 0.0])
    else:
        s = np.sin(theta / 2)

        X = np.array([[0, 1],
                      [1, 0]], dtype=DTYPE)

        Y = np.array([[0, -1j],
                      [1j, 0]], dtype=DTYPE)

        Z = np.array([[1, 0],
                      [0, -1]], dtype=DTYPE)

        nx = np.real((1j / (2 * s)) * np.trace(X @ u))
        ny = np.real((1j / (2 * s)) * np.trace(Y @ u))
        nz = np.real((1j / (2 * s)) * np.trace(Z @ u))

        n = np.array([nx, ny, nz])
        n /= np.linalg.norm(n)

    # ----- Step 5: Return Bloch object -----
    b = Bloch()
    b.alpha = alpha
    b.n = n
    b.theta = theta

    return b

# def n1n2n1_angles(b: Bloch) -> tuple[float, float, float, float]:
#     """Factor the rotation part of a unitary (given as its Bloch form `b`) as
#         u = e^{i global_phase} * Rn1(alpha) * Rn2(beta) * Rn1(gamma)

#     where Ra(angle) is a rotation by `angle` about axis a, and {a1, a2, a3} is
#     the orthonormal frame defined above. Returns (alpha, beta, gamma, global_phase).
#     """
#     # TODO(student): implement using the steps above.
#     raise NotImplementedError("n1n2n1_angles is not implemented yet")
def n1n2n1_angles(b: Bloch):
    phi = b.theta

    x = np.dot(b.n, a1)
    y = np.dot(b.n, a2)
    z = np.dot(b.n, a3)

    cos_beta = np.sqrt(
        np.cos(phi)**2 +
        (x * np.sin(phi))**2
    )
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)

    S = np.arctan2(
        x * np.sin(phi),
        np.cos(phi)
    )

    D = np.arctan2(z, y)

    gamma = 0.5 * (S + D)
    alpha = 0.5 * (S - D)

    return alpha, beta, gamma, b.alpha


def approx_angle_with_tolerance(angle: float, tolerance: float) -> int:

    target = angle % TWO_PI

    k = 0

    while True:

        candidate = (k * LAMBDA_PI) % TWO_PI

        diff = abs(candidate - target)
        diff = min(diff, TWO_PI - diff)

        if diff <= tolerance:
            return k

        k += 1


def decompose_2x2(u: np.ndarray, tolerance: float) -> tuple[int, int, int]:

    bloch = to_bloch(u)

    alpha, beta, gamma, _ = n1n2n1_angles(bloch)

    k = approx_angle_with_tolerance(alpha, tolerance)
    l = approx_angle_with_tolerance(beta, tolerance)
    m = approx_angle_with_tolerance(gamma, tolerance)

    return k, l, m
