# Methodology

This document defines every metric the report publishes, states why it is
defined that way, and lists the assumptions that would have to hold for the
numbers to mean what they appear to mean. It is written so that a reader who
disagrees with a definition can find it, change it in one place, and rerun the
pipeline.

The dataset is synthetic. It is produced by `src/generate_data.py` from a fixed
seed and describes no real firm, client, engagement or person. Where the
generator makes a behavioural assumption — fixed-fee work overruns, small
engagements struggle to cover mobilisation cost — that assumption is stated in
the generator's module docstring, not here. This document is about the
arithmetic applied to the data once it exists.

---

## 1. The shape of the data

Five tables, all derived from one seed:

| Table | Grain | Rows |
| --- | --- | --- |
| `consultants.csv` | one row per person | 90 |
| `clients.csv` | one row per client | 60 |
| `engagements.csv` | one row per engagement | 200 |
| `timesheets.csv` | one row per engagement × consultant × month | 11,350 |
| `billing.csv` | one row per engagement | 200 |

The portfolio spans three calendar years of delivery. An engagement that starts
in 2024 and finishes in 2025 has hours in both, so an engagement is not the same
thing as a year of work. `src/analysis.py` therefore works at two grains:

- The **engagement fact table**, one row per engagement, used for anything that
  describes the work itself — client concentration, contract type, service line.
- The **engagement-year fact table**, one row per engagement per year, used for
  anything that describes a period. The headline fiscal year is read from this
  grain.

### Why money is split across years by hours

An engagement's money is recorded once, against the engagement. To report a
single year, that money has to be spread over the years the work took.
Percentage-of-completion is used: each year carries the share of the
engagement's hours that were worked in it, and every money column — rate-card
value, discount, unbilled time, invoice, write-off, collection, both cost lines
and margin — is scaled by that same share. The consequence is that the year rows
for an engagement sum back exactly to its engagement-level row, so the waterfall
identity holds at both grains and a year can be isolated without restating the
portfolio.

The alternative — recognising revenue on invoice date — would put a three-year
engagement's entire margin in the year it was billed, which would make the
annual comparison a report about invoicing policy rather than about delivery.
Neither is right in every context. This one is stated, applied consistently, and
reversible in one function.

---

## 2. The commercial funnel

The report's central claim is that the distance between what work is worth and
what the firm keeps is not one problem but five, and that they need different
remedies. The funnel makes that concrete.

### Standard value

`standard_value = delivered hours × the rate card rate of the grade that delivered them`

Delivered hours come from the timesheets and are valued at the standard rate
attached to each hour. This is what the work was worth before any commercial
conversation: no discount, no scope creep, no write-off. It is the denominator
for realisation and the opening bar of the waterfall.

Using delivered hours rather than budgeted hours matters. Valuing budgeted hours
would hide overrun entirely, because an engagement that took twice as long as
planned would look the same as one that finished on plan.

### Billed amount

`billed_amount = standard_value − discount_amount − unbilled_value`

What was actually invoiced. Two different things are removed.

**Discount** is a commercial decision made before the work is delivered: the
rate card was never going to be charged. It is visible on the engagement record.

**Unbilled value** is time that was worked, was worth something at the rate card,
and was never invoiced at all. It is the residual between what the timesheets say
was delivered and what the engagement's fee could support.

### Collected amount

`collected_amount = billed_amount − write_off_amount`

What the client actually paid. This is net revenue, and it is the denominator for
margin. Write-offs are the difference between the invoice and the cash: a
rejected timesheet line, a disputed disbursement, a goodwill credit agreed at the
end of a relationship.

### The three derived rates

| Rate | Definition | Question it answers |
| --- | --- | --- |
| Realisation | `billed / standard_value` | How much of the rate card survived pricing and delivery? |
| Collection | `collected / billed` | How much of the invoice survived the client's payment process? |
| Margin rate | `margin / collected` | How much of each collected dollar is left? |

All three are computed from summed totals, never as an average of per-engagement
ratios. Averages of ratios weight a $20,000 engagement the same as a $2,000,000
one, which is exactly the wrong weighting when the question is where the firm's
money went.

### Margin

`margin = standard_value − discount − unbilled − write-off − direct delivery cost − consultant cost`

Margin is defined as the **waterfall residual** rather than re-derived from
collections. The two are algebraically identical — subtracting the deductions
from the rate card is the same as subtracting cost from collections — but
defining it as the residual means the published waterfall adds up exactly rather
than drifting by a cent per engagement through independent rounding.
`waterfall_reconciles()` asserts the identity, and the test suite asserts it
again on hand-computed inputs. On the published FY2025 data the residual is
under 6 × 10⁻¹¹ of a dollar.

Consultant cost is the delivering grade's cost rate applied to the hours worked.
Direct delivery cost is everything else the engagement consumed — travel,
subcontractors, licences, research — carried as a share of rate-card value.

---

## 3. Realisation versus collection

The two are frequently collapsed into one number called "revenue leakage". They
are different failures with different owners and different remedies, and
separating them is the point of publishing both.

**Realisation is lost before the invoice exists.** Nobody outside the firm has to
agree to it. It is scope that grew without a change order, a fee that was
discounted to win work, time that was worked and never submitted. The firm
controls all of it. In FY2025 it costs $8.9M — 19.9% of the rate card — and 78%
of that is discounts and unbilled time.

**Collection is lost after the invoice exists.** It requires the client's
agreement and their accounts payable process. In FY2025 it costs $1.9M, 5.1% of
what was billed. That is a different-sized problem and it is not where the margin
is.

### Why unbilled time is the quiet one

Discounts are negotiated, recorded and reviewed. Write-offs surface on an
ageing report. Unbilled time does not appear anywhere: the hours exist in the
timesheet system, the invoice is correct against the fee, and the variance is
absorbed silently by the engagement's margin.

At $2.68M in FY2025 it is 6.0% of the rate card — larger than write-offs, and
the only one of the three that no report currently names. It is also the most
concentrated: 8.7% of rate-card value on fixed-fee work against 3.2% on
time-and-materials. That gap is the signature of a contract type where the fee
is fixed but the hours are not, and where extra hours therefore have nowhere to
go.

---

## 4. Utilisation

`utilization = billable hours / available hours`

Available hours are the chargeable hours a person makes available inside the
analysis window. Someone who joined mid-window accrues pro rata, so a June
joiner does not carry a full year of denominator.

**Utilisation is recomputed from summed hours at every level of aggregation**,
never averaged across people or across grades. Averaging per-person rates gives
a part-year joiner the same weight as a full year of availability, which
overstates the firm. The firm rate is total billable hours over total available
hours.

The staffing model that produces the timesheets is capacity-aware: an engagement
can only book the hours a person has left, in total and in each month the
engagement touches. This matters more than it sounds. A staffing draw that
ignores capacity produces a roster where a handful of people are booked at four
times their availability while the rest sit idle — the aggregate utilisation
still looks plausible, and every per-person figure in the interactive view is
nonsense.

---

## 5. Client tiers

Clients are ranked by margin and assigned a tier by where their **cumulative**
margin lands:

- **Tier A** — up to 50% of firm margin.
- **Tier B** — up to 80%.
- **Tier C** — the remainder.

Both cuts are inclusive, and the largest client is placed in tier A whatever its
own share, so tier A is never empty.

Cumulative cuts, rather than ranking into thirds, are used because the question
is concentration: how few clients carry most of the firm's profit. Tier A in this
portfolio is 9 clients carrying 46.6% of revenue and 49.7% of margin. Twenty-three
of sixty clients carry 80% of margin between them.

Tiers are cut on margin share rather than revenue share deliberately. A client
can be large and unprofitable; ranking by revenue would put a loss-making account
in tier A and hide it.

---

## 6. Overrun

`overrun_ratio = delivered hours / budgeted hours`

Overrun is measured in hours, not money, so that it is independent of the rate
card and the fee. An engagement at 1.10x delivered 10% more hours than it was
planned to.

The relationship between overrun and margin is reported two ways. The overrun
bands sort engagements into quintiles and show the mean margin rate in each,
which is robust and readable. The realisation deciles do the same for
realisation, using the **median** margin rate within each decile rather than a
fitted line.

The medians are used because the relationship is monotone but not linear and
strongly heteroscedastic: a handful of severely loss-making engagements sit in
the low-realisation tail and would dominate any straight line drawn through
them. A fitted line through those points implies a specific margin at zero
realisation, which is a claim the data cannot support.

---

## 7. The sensitivity, and what closing the gap is worth

Every recovery figure in the report — the sensitivity table and all three
recommendations — comes from one function:

```
incremental margin = rate-card value of the group
                   × points of realisation recovered / 100
                   × the group's collection rate
```

Three assumptions are baked in, and they are the assumptions a reader should
argue with:

1. **Recovered billing is collected at the group's existing collection rate.**
   Not 100%, and not the firm's best rate — the rate those engagements already
   achieve.
2. **Recovered billing carries no incremental delivery cost.** The hours have
   already been worked. This is what makes realisation a cheaper lever than
   winning new work, and it is also the assumption that fails if "recovering
   realisation" turns out to mean working more hours.
3. **No new work is assumed.** The estimate is what is already earned and not
   billed, not a growth forecast.

Pushing realisation harder has a cost the estimate does not price: a client
asked to pay for time already delivered may slow payment on the rest. The
recommendations are therefore sized as opportunities, not as forecasts, and the
report says so on the page.

**The three recommendations are not additive.** They act on overlapping
populations — a small fixed-fee engagement with weak realisation is counted by
all three. Each figure is individually defensible; the sum is not a plan.

---

## 8. Limitations

**The data is synthetic.** Every number in the report describes a generated
portfolio. The structure is modelled on how consulting firms behave; the
magnitudes are assumptions. Nothing here is an observation about any real firm.

**Assignment by grade is an allocation, not a measurement.** Billing is recorded
against an engagement, not against a grade. The by-grade view spreads each
engagement's billing across grades in proportion to the rate-card value each
grade contributed. It shows which grades staff the engagements where realisation
is won or lost. It does not show that any grade prices differently, and it
should not be read that way.

**Money is allocated across years by hours.** Percentage-of-completion is a
reasonable approximation for a portfolio of this shape, but it is an
approximation. An engagement's cash does not necessarily follow its hours within
a year, and a portfolio with large milestone payments would break the
assumption.

**The sensitivity is static.** It holds collection rates and cost shares fixed
while realisation moves. In practice those interact.

**Unbilled value is a residual, not a ledger entry.** The generator derives it
from the gap between delivered value and the engagement's fee. In a real firm it
would come from a time-and-expense system that flags write-downs, which would
make it measurable rather than inferred.

**Ninety consultants is a small firm.** Per-person utilisation is noisy, and the
lowest observed figure is as much a staffing artefact as a productivity signal.
The grade-level and firm-level figures are the ones worth acting on.

**One year is not a trend.** Three years of synthetic history are reported, which
is enough to show the direction of margin rate and not enough to establish one.
