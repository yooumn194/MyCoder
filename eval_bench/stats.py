"""Sample statistics shared by the evaluation reports.

This rule — mean, population standard deviation, four decimals, ``None`` when
there is nothing to measure — used to be written out twice inside ``matrix.py``
(the per-field metric and the pass-rate block). The SWE-bench repeat report
needs it a third time, and three copies of a rounding rule is how two reports
end up disagreeing about the same run. Kept in its own module rather than in
either report so that neither imports the other's private helper.
"""

from __future__ import annotations

import statistics
from typing import Iterable

# The reports print four decimals, and the matrix tests compare these floats for
# equality, so the rounding is part of the output contract rather than a
# formatting choice.
PRECISION = 4


def summarize(values: Iterable[float]) -> dict:
    """Mean and population standard deviation over one sample.

    Population, not sample, deviation: the repeats are the whole set being
    described rather than a draw from a larger population, which is also what
    the ablation matrix has always reported.

    ``stddev`` is ``None`` for a single observation. ``pstdev`` returns ``0.0``
    there, and perfect stability is a claim only two or more observations can
    support — reporting ``0.0`` from one run is the same error as publishing an
    unmeasured quality score as ``0.5``. ``mean`` is still reported for n == 1,
    because that one value genuinely is the mean of what was measured.
    """
    numbers = [float(value) for value in values]
    if not numbers:
        return {"mean": None, "stddev": None, "n": 0}
    return {
        "mean": round(statistics.fmean(numbers), PRECISION),
        "stddev": round(statistics.pstdev(numbers), PRECISION) if len(numbers) > 1 else None,
        "n": len(numbers),
    }
