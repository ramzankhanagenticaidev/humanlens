# Third-Party Licenses

HumanLens itself is MIT licensed (see `LICENSE`). It depends on the following
open-source packages, all under permissive licenses that allow commercial use
and redistribution. This is a summary for convenience — always verify against
each project's own repository before shipping a commercial product, and check
the installed version's exact license file (`pip show <package>` or the
package's `LICENSE`/`COPYING` file in `site-packages`) since terms can change
between releases.

| Package | License |
|---|---|
| FastAPI | MIT |
| Uvicorn | BSD-3-Clause |
| Pydantic | MIT |
| Streamlit | Apache-2.0 |
| Plotly | MIT |
| Pandas | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| SciPy | BSD-3-Clause |
| NumPy | BSD-3-Clause |
| NLTK | Apache-2.0 |
| textstat | MIT |
| joblib | BSD-3-Clause |
| Requests | Apache-2.0 |
| pytest | MIT |
| httpx | BSD-3-Clause |
| python-dotenv | BSD-3-Clause |

None of these licenses require you to open-source your own product or pay
royalties. The BSD/MIT licenses generally only require keeping the original
copyright notice somewhere in your distribution (this file serves that
purpose). Apache-2.0 (Streamlit, NLTK, Requests) additionally requires
preserving any `NOTICE` file the project ships and stating significant
changes if you modify their source directly (not applicable here — HumanLens
only imports these as libraries, it doesn't vendor or modify their code).

**Not legal advice.** If you're shipping this as a paid product at scale,
have a lawyer confirm license compliance and check for any updates to these
terms in newer releases of each dependency.

## NLTK data

`nltk.download('punkt_tab')` downloads Punkt tokenizer models, which are
separately licensed (Punkt itself is public-domain / freely redistributable
research data). No additional restrictions apply for commercial use.
