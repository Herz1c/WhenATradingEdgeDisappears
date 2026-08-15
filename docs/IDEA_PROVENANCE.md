# Where my research ideas appear in the record

I wrote this document to connect my major research decisions to the best local evidence that
survives. It shows which ideas appear in earlier records and which rely on my present
declaration.

## How I label the evidence

- **L1 - earlier decision record:** one of my local decision entries or audit reports states
  the methodological rule before the final TCN result.
- **L2 - development sequence:** local commits or dated artifacts show the idea being
  implemented and revised during the project.
- **L3 - my present declaration:** I state that the idea was mine, but I do not have an
  earlier independently verifiable record for that exact decision.

None of these labels is an external timestamp. I use them so that a reviewer can distinguish
earlier corroboration from my present statement.

## Idea-by-idea record

| Idea or decision I originated | Surviving corroboration | Evidence level |
|---|---|---|
| Study whether cross-venue BTC information improves five-minute prediction-market prices | Initial strategy/model history and the multi-source architecture | L2 |
| Treat recorder receive time as the causal availability clock | My April 23 `delta_to_strike Canonical Reconnection` and April 28 `Canonical Event Base and Asof Join Contract` entries in `decision_log.md` | L1 |
| Preserve invalid data with explicit flags instead of silently dropping it | My April 24 audit stop condition and April 25 market-exclusion decisions | L1 |
| Concentrate sampling near market close without using post-close information | My April 23 `Phase 3 Sampling Policy Freeze` entry | L1 |
| Require honest baselines and treat calibration leakage as a methodological failure | My May 12 `quant_audit_findings.md` and completion report | L1 |
| Stop an audit instead of waiving a failed contract | My April 24 `Phase 1-3 Audit Stop Condition` entry | L1 |
| Require causal order-book replay, cancel/replace ordering, fee economics, and inventory accounting | My April 26 Phase 5 simulator and hardening decisions | L1 |
| Treat execution assumptions as audit findings, not implementation waivers | My April 26 `Phase 5 Audit Interpretation of V1 Assumptions` entry | L1 |
| Refuse deployment until paper-trading, markout, safety, and audit gates pass | My April 28 model/evaluation/live-infrastructure decision | L1 |
| Add bias-freshness and source-health refusal gates to the shadow bot | May 26 private-source commits and July shadow decision denominators | L2 |
| Correct for the searched strategy family with DSR/White-style anti-snooping tests | Locally dated July 11 audit artifact and my present declaration | L2/L3 |
| Treat repeated timesteps as dependent and use a paired market-cluster interval | My present declaration and the public deterministic reproduction | L3 |
| Require adverse execution tests to hold the decision set fixed | My earlier execution-contract decisions and final audit specification | L1/L3 |
| Publish the failure analysis and accept "no demonstrated edge" | My signed declaration and the final public claim register | L3 |

## Why my earlier records matter

I did not add the causal and adversarial approach only after seeing the final result. My
surviving April and May documents already require receive-time joins, stop audits when a
contract fails, reject misleading baselines, flag calibration leakage, enumerate execution
requirements, and prohibit deployment without safety gates. The later source-time failure
violated rules I had already written down; it was not the origin of those rules.

## Boundary of the surviving record

The exported commit sequence does not cover the July/August phase. That period is represented
by dated artifacts and my present declaration rather than commit history. In future work, I
will preserve signed specifications, prompt records, and verified commits alongside the
technical artifacts.
