# Dataset Construction and Preprocessing

## 1. Construct the original 25 × 17 weekly revenue panel

Implemented by [00a_build_panel.py](../00a_build_panel.py).

### 1.1 Aggregate the raw records

- **Source:** UPC-level movement CSVs downloaded via the **Movement .csv File** links on the [official Dominick's dataset page](https://www.chicagobooth.edu/research/kilts/research-data/dominicks). Download `w<category>.csv` for each of the 17 category codes below and place the files in `realdata/data/dominick_raw/`, or pass their directory to `00a_build_panel.py` with `--raw-dir`. Each row records one product's weekly sales at one store.
- **Category:** determined by the source file `w<category>.csv`; all valid UPC records in that file contribute to its category.
- **Sample:** the following stores, categories, and ordering are fixed in the builder.

| Dimension | Included values, in panel order |
| --- | --- |
| 25 stores | `136, 72, 67, 33, 84, 32, 126, 112, 129, 2, 94, 53, 105, 121, 119, 47, 109, 9, 80, 59, 137, 68, 93, 102, 14` |
| 17 categories | `ana, bjc, che, coo, cra, did, fec, frd, fre, frj, fsf, gro, lnd, oat, sna, soa, tpa` |
| 122 weeks | Source weeks 278–399, indexed as panel weeks $t=1,\ldots,122$; week starts January 5, 1995–May 1, 1997 |

### 1.2 Clean records and calculate revenue

Retain integer `STORE` and `WEEK` values within the fixed sample, then apply these rules. Field meanings follow the [Dominick's data manual, Part 4](https://www.chicagobooth.edu/research/kilts/research-data/-/media/enterprise/centers/kilts/datasets/dominicks-dataset/dominicks-manual-and-codebook_kiltscenter.aspx#page=10).

| Field | Meaning | Valid-record rule |
| --- | --- | --- |
| `MOVE` | Number of individual items sold | Finite and ≥ 0 |
| `QTY` | Number of items covered by the quoted price | Finite and > 0 |
| `PRICE` | Total retail price for `QTY` items, in dollars | Finite and ≥ 0 when `MOVE > 0` |
| `OK` | Source validity flag | Equal to 1 |

For a valid UPC record $r$, unit price is $\mathrm{PRICE}_r/\mathrm{QTY}_r$, and revenue is

$$
v_r=
\begin{cases}
0, & \mathrm{MOVE}_r=0,\\
\mathrm{MOVE}_r\,\dfrac{\mathrm{PRICE}_r}{\mathrm{QTY}_r}, & \mathrm{MOVE}_r>0.
\end{cases}
$$

- Example: `MOVE = 12`, `PRICE = 3`, `QTY = 2` gives revenue of **18 dollars**.
- A valid zero-sales record contributes zero even when its price is missing.

### 1.3 Aggregate into the panel

Let $\mathcal V_{ijt}$ contain all retained UPC records for store $i$, category $j$, and panel week $t$. Weekly category revenue is their **sum**:

$$
R_{ijt}=
\begin{cases}
\displaystyle\sum_{r\in\mathcal V_{ijt}}v_r, & \mathcal V_{ijt}\ne\varnothing,\\
\mathrm{NaN}, & \mathcal V_{ijt}=\varnothing.
\end{cases}
$$

- Each week gives a **25-store × 17-category** matrix; stacking 122 weeks produces the **25 × 122 × 17** array, ordered as store × week × category.
- A cell is observed ($O_{ijt}=1$) whenever at least one valid UPC record remains. Observed zero revenue stays zero; a cell with no valid records stays missing.
- Save dollar revenues, the observed mask, coordinates, and record counts in `raw_panel.npz`.

## 2. Preprocess the revenue panel

Implemented by [00b_preprocess_panel.py](../00b_preprocess_panel.py), in the order below.

### 2.1 Log transformation

$$
y_{ijt}=\log(1+R_{ijt}).
$$

Zero revenue becomes zero; missing cells remain missing at this step.

### 2.2 Trend, seasonality, and holiday adjustment

Fit the following regression by OLS **separately for each store–category series** $(i,j)$ and each stage $s$:

$$
\begin{aligned}
y_{ijt}
={}&\alpha_{ij}^{(s)}+\beta_{ij}^{(s)}u_t\\
&+\sum_{k=1}^{3}\left[
\gamma_{ijk}^{(s)}\sin\left(\frac{2\pi k u_t}{52}\right)
+\delta_{ijk}^{(s)}\cos\left(\frac{2\pi k u_t}{52}\right)
\right]\\
&+\sum_{h=1}^{7}\eta_{ijh}^{(s)}H_{ht}+\varepsilon_{ijt}^{(s)}.
\end{aligned}
$$

Here $u_t=t-1$ and $H_{ht}$ indicates holiday $h$. The regression contains 15 terms:

| Component | Regressors | Count |
| --- | --- | --- |
| Intercept | $1$ | 1 |
| Trend | $u_t$ | 1 |
| Seasonality | $\sin(2\pi k u_t/52)$, $\cos(2\pi k u_t/52)$; $k=1,2,3$ | 6 |
| Holidays | One indicator per holiday | 7 |

- **Holidays:** New Year's Day, Easter Sunday, Memorial Day, Independence Day, Labor Day, Thanksgiving, Christmas.
- **Holiday week:** indicator = 1 if the holiday falls within `[week start, week start + 6 days]`.
- **Separate CV/test estimation:** Use only originally observed (non-missing) values in each stage's initial training set: weeks 1–82 for CV and weeks 1–102 for the final test. Estimate the regression effects independently for the two stages.
- **Apply to all weeks:** Apply each stage's fitted coefficients to all 122 weeks of the same store–category series. Subtract the fitted effects from log revenue to obtain separate adjusted response series for CV and the final test.

Let $\widehat y_{ijt}^{(s)}$ be the fitted value from the regression above. The adjusted series is

$$
e_{ijt}^{(s)}=y_{ijt}-\widehat y_{ijt}^{(s)}.
$$

### 2.3 Fill missing residuals

After removing trend, seasonal, and holiday effects from log revenue, fill missing residuals by linear interpolation.

### 2.4 Centering and standardization

Center each store–category series separately and standardize each category using a standard deviation pooled across stores and weeks. Preprocessing parameters are estimated within each stage's fit period and kept fixed as its model window rolls.

## 3. Estimation and forecast windows

All ranges are inclusive panel-week indices.

| Stage $s$ | Preprocessing fit period | Model windows | Forecast weeks | Scored occasions |
| --- | --- | --- | --- | --- |
| Validation (CV) | 1–82 ($q_s=82$) | 1–82, …, 20–101 | 83–102 | 17 |
| Test | 1–102 ($q_s=102$) | 1–102, …, 20–121 | 103–122 | 20 |
| Full sample | 1–122 ($q_s=122$) | 1–122 | None; descriptive fit | — |

- Each rolling window advances by one week and predicts the next week.
- **CV weeks 93–94:** excluded from scoring because some original observations are missing.
- **CV week 95:** excluded because its adjusted residuals were used to fill missing residuals in week 94. Using the filled week 94 to predict week 95 would use future information.

## 4. Arrays entering the model

All arrays below are in `processed.npz`; axis order is **store × week × category**.

| Arrays | Shape / type | Use |
| --- | --- | --- |
| `validation_standardized`, `test_standardized`, `full_standardized` | 25 × 122 × 17; finite float64 | Stage-specific standardized residuals $z^{(s)}$ |
| `validation_actual`, `test_actual` | 25 × 20 × 17; float64 | Forecast targets on the same standardized scale |
| `validation_score_mask`, `test_score_mask` | 25 × 20 × 17; bool | Cells included in scoring |
| `original_observed_mask`, `imputation_mask` | 25 × 122 × 17; bool | Originally observed versus filled entries |

- Each stage uses its own standardized array.
- VAR(1) training pairs are $(z_{i,t-1}^{(s)},z_{it}^{(s)})$, with 17 categories per vector, within the selected window.
- The training runner slices these saved arrays; it does not repeat preprocessing.
- Reported forecast MSE uses this standardized residual scale.
