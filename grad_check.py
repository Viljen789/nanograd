"""
grad_check.py  --  a generic numerical gradient checker (central finite differences).

WHAT THIS IS
    Your ORACLE for Phases 1-3. It tells you whether the gradients your autograd
    engine computes are *mathematically correct*, with no reference implementation
    and no knowledge of how you built things.

WHY IT WORKS AGAINST ANY INTERFACE YOU INVENT
    It only ever touches plain Python floats / numpy arrays. It never imports, names,
    or assumes anything about your engine's objects. You connect your engine to it by
    writing a thin adapter (a few lines) that speaks only in plain numbers:

        f(x) -> float   : take a point x (a numpy array of floats), build a FRESH graph
                          from those numbers, run YOUR forward pass, and return the
                          final scalar as a plain float.
        analytic_grad   : after you run YOUR backward pass at that same point x, collect
                          the gradients your engine produced into a numpy array with the
                          SAME shape as x.

    The checker estimates the gradient numerically and compares it to yours. Because the
    boundary is just numbers, the same file works for your scalar engine AND, later, your
    tensor library -- you never have to change it.

THE MATH
    A derivative is the limit of a slope. We approximate it without limits using a
    symmetric ("central") difference, perturbing ONE input at a time and holding the
    rest fixed:

        df/dx_i  ~=  ( f(x + eps * e_i) - f(x - eps * e_i) ) / (2 * eps)

    The symmetric form has O(eps^2) truncation error -- far better than the one-sided
    ( f(x+eps) - f(x) ) / eps, which is only O(eps). Use float64. eps ~ 1e-6 is the
    sweet spot in float64: too big -> truncation error dominates; too small -> floating
    point roundoff dominates.

READING THE RESULT
    A correct backward pass agrees to a tiny relative error (<= ~1e-5). A wrong local
    rule, or gradients overwritten instead of accumulated, blows the error up to O(0.1)
    or more -- impossible to miss. A near-miss (1e-3..1e-2) usually means a kink, eps,
    a float32 issue, or catastrophic cancellation (see below) -- not a logic bug.

CATASTROPHIC CANCELLATION (a way a CORRECT gradient can spuriously FAIL)
    The numerical estimate subtracts f(x+eps) and f(x-eps). If the function's VALUE is
    huge relative to how much it changes (e.g. a loss like 1e6 + small_signal), those two
    numbers are nearly equal and the subtraction loses precision. This won't bite the
    small values of Phases 1-3, but if a later check fails by ~1e-4 on code you trust,
    shrink the function's magnitude (or subtract its constant baseline) before checking.
"""

import numpy as np


def numerical_gradient(f, x, eps=1e-6):
    """Central-difference estimate of grad f at x.

    f   : callable  (numpy array shaped like x) -> float
    x   : numpy array of float64, any shape
    eps : perturbation size
    returns: numpy array shaped like x
    """
    x = np.asarray(x, dtype=np.float64).copy()
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        original = x[idx]
        x[idx] = original + eps
        f_plus = float(f(x.copy()))
        x[idx] = original - eps
        f_minus = float(f(x.copy()))
        x[idx] = original  # restore before moving to the next coordinate
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
    return grad


def grad_check(f, x, analytic_grad, eps=1e-6, rtol=1e-5, atol=1e-8, verbose=True):
    """Compare YOUR analytic gradient to a numerical one.

    f             : callable (numpy array like x) -> float   -- your wrapped forward pass
    x             : numpy array of float64                   -- the point to check at
    analytic_grad : numpy array shaped like x                -- gradients YOUR engine produced
    eps           : perturbation size for the numerical estimate
    rtol, atol    : pass elementwise if |a - n| <= atol + rtol*max(|a|,|n|)  (numpy.allclose form;
                    the atol term keeps near-zero gradients from raising spurious relative error)
    verbose       : print a short report

    Pick x RANDOM and non-degenerate: not all zeros, and away from kinks (ReLU/abs/max
    corners) where the true derivative jumps and finite differences is meaningless.

    Returns: (passed: bool, max_relative_error: float, numerical_grad: np.ndarray)
    """
    x = np.asarray(x, dtype=np.float64).copy()
    analytic = np.asarray(analytic_grad, dtype=np.float64)
    if analytic.shape != x.shape:
        raise ValueError(
            f"analytic_grad shape {analytic.shape} != x shape {x.shape}. "
            "Your analytic gradient must line up element-for-element with x."
        )

    numerical = numerical_gradient(f, x, eps=eps)

    abs_err = np.abs(analytic - numerical)
    scale = np.maximum(np.abs(analytic), np.abs(numerical))
    ok = abs_err <= (atol + rtol * scale)
    passed = bool(np.all(ok))

    # Headline relative error is measured only over elements big enough for a *relative* measure to
    # mean something. Near-zero-gradient elements pass on the atol term; including them would inflate
    # the reported number (a true-zero gradient with tiny roundoff has relative error ~1) even though
    # the verdict is correct. The PASS/FAIL decision above already used the full allclose criterion.
    significant = scale > atol
    rel_err = np.where(significant, abs_err / np.maximum(scale, 1e-12), 0.0)
    max_rel = float(np.max(rel_err))

    if verbose:
        worst = np.unravel_index(int(np.argmax(rel_err)), rel_err.shape)
        print(f"grad_check: {'PASS' if passed else 'FAIL'}   "
              f"max relative error = {max_rel:.2e}   (rtol {rtol:.0e}, atol {atol:.0e})")
        print(f"  worst element {worst}:   "
              f"analytic = {float(analytic[worst]):+.6e}   "
              f"numerical = {float(numerical[worst]):+.6e}")
        if not passed:
            print("  Suspect, in order:")
            print("   1. a wrong local-derivative rule for one operation;")
            print("   2. gradient OVERWRITTEN instead of ACCUMULATED where a value")
            print("      feeds more than one consumer (the classic bug);")
            print("   3. you checked at a kink (ReLU/abs/max corner) or at x = 0;")
            print("   4. float32 sneaking in somewhere -- everything must be float64.")
            print("  Debug by shrinking the graph until it PASSES, then add ops back one")
            print("  at a time; the op that breaks it is the one with the wrong rule.")

    return passed, max_rel, numerical


if __name__ == "__main__":
    # "Trust your oracle": validate the checker itself against derivatives you know by hand.
    # Run me with:  python grad_check.py
    rng = np.random.default_rng(0)

    print("1) f(x) = x^3 - 2x      (true f' = 3x^2 - 2)   -- should PASS")
    x = rng.uniform(-2, 2, size=(1,))
    grad_check(lambda v: v[0] ** 3 - 2 * v[0], x, np.array([3 * x[0] ** 2 - 2]))

    print("\n2) f(x) = 0.5 * sum(x_i^2)   (true grad = x)   -- should PASS")
    x = rng.uniform(-3, 3, size=(5,))
    grad_check(lambda v: 0.5 * np.sum(v ** 2), x, x.copy())

    print("\n3) WRONG analytic gradient on purpose          -- should FAIL (proves the checker bites)")
    x = rng.uniform(-1, 1, size=(3,))
    grad_check(lambda v: np.sum(np.sin(v)), x, np.cos(x) + 0.1)
