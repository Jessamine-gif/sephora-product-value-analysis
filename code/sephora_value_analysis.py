import os, glob, warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

#Settings
DATA_DIR = "/Users/yingjingjiang/Downloads/PythonProject1/sophere_reviews"
OUT_DIR  = "/Users/yingjingjiang/Downloads/PythonProject1/separate_figures"

MIN_BRAND_REVIEWS = 30
WINSOR_Q = (0.01, 0.99)
SEED = 42
rng = np.random.default_rng(SEED)

# Color scheme
BARBIE_PINK = "#E0218A"
LIGHT_PINK  = "#FFB6D9"
BLACK       = "#000000"
CHARCOAL    = "#2B2B2B"
ACCENT      = "#7A3E9D"

# Gradient colors
GRADIENT_COLORS = ['#E0218A', '#FF6B9D', '#FFB6D9', '#D4A5C4', '#9B90C2']

sns.set_theme(
    context="notebook",
    style="whitegrid",
    rc={
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "grid.color": LIGHT_PINK,
        "axes.labelweight": "bold",
        "axes.edgecolor": CHARCOAL,
        "text.color": CHARCOAL,
        "axes.labelcolor": CHARCOAL,
        "xtick.color": CHARCOAL,
        "ytick.color": CHARCOAL,
    }
)

def style_axes(ax):
    """Apply my custom styling to plot axes"""
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, linestyle="--", alpha=0.35, color=LIGHT_PINK, linewidth=0.8)
    for s in ["top","right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(CHARCOAL)
    ax.spines["bottom"].set_color(CHARCOAL)

#Helper functions
def pick_col(cols, candidates):
    """Find column name from possible variations"""
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    for c in cols:
        cl = c.lower()
        for cand in candidates:
            if cand in cl:
                return c
    return None

def winsorize(s, lower_q=0.01, upper_q=0.99):
    """Cap extreme values at percentiles"""
    lo, hi = np.nanquantile(s, [lower_q, upper_q])
    return s.clip(lo, hi)

def try_import_adjusttext():
    """Try to import adjustText for label placement"""
    try:
        from adjustText import adjust_text
        return adjust_text
    except Exception:
        return None

# Load data
print("Loading data...")
review_files = glob.glob(os.path.join(DATA_DIR, "reviews_*.csv"))
if not review_files:
    raise FileNotFoundError("No review files found: reviews_*.csv")
reviews = pd.concat((pd.read_csv(f, low_memory=False) for f in review_files), ignore_index=True)

products_path = os.path.join(DATA_DIR, "product_info.csv")
if not os.path.exists(products_path):
    raise FileNotFoundError(f"Product info not found: {products_path}")
products = pd.read_csv(products_path, low_memory=False)

#Merge data & auto-detect columns
# Find product ID columns in both dataframes
pid_reviews  = pick_col(reviews.columns,  {"product_id","id","item_id","productid"})
pid_products = pick_col(products.columns, {"product_id","id","item_id","productid"})
if pid_reviews is None or pid_products is None:
    raise RuntimeError(f"Product ID not found")

# Merge reviews with product info
df = reviews.merge(products, left_on=pid_reviews, right_on=pid_products, how="left")

# Find important columns
col_rating   = pick_col(df.columns, {"rating","stars","review_rating","rating_value"})
col_price    = pick_col(df.columns, {"price","list_price","current_price","original_price"})
col_brand    = pick_col(df.columns, {"brand","brand_name","product_brand"})
col_category = pick_col(df.columns, {"category","categories","product_category"})

# Rename to standard names
rename_map = {}
if col_rating:   rename_map[col_rating]   = "rating"
if col_price:    rename_map[col_price]    = "price"
if col_brand:    rename_map[col_brand]    = "brand"
if col_category: rename_map[col_category] = "category"
df = df.rename(columns=rename_map)

# Convert to numeric
for c in ["rating","price"]:
    df[c] = pd.to_numeric(df.get(c), errors="coerce")

# Check required columns exist
need_cols = ["rating","price","brand"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

# Clean data
df = df.dropna(subset=need_cols)
df = df[df["price"] > 0]
df = df[df["rating"].between(1, 5)]
df["price"] = winsorize(df["price"], WINSOR_Q[0], WINSOR_Q[1])
df["brand"] = df["brand"].astype(str).str.strip().str.title()

# Calculate brand-level statistics
brand_stats = (
    df.groupby("brand")
      .agg(avg_rating=("rating","mean"),
           avg_price =("price","mean"),
           reviews_n =("rating","size"))
      .reset_index()
)
# Keep only brands with enough reviews
brand_stats = brand_stats[brand_stats["reviews_n"] >= MIN_BRAND_REVIEWS].copy()

# Calculate value score and z-scores
brand_stats["value"] = brand_stats["avg_rating"] / brand_stats["avg_price"]
brand_stats["z_rating"] = (brand_stats["avg_rating"] - brand_stats["avg_rating"].mean()) / brand_stats["avg_rating"].std(ddof=0)
brand_stats["z_price"]  = (brand_stats["avg_price"]  - brand_stats["avg_price"].mean())  / brand_stats["avg_price"].std(ddof=0)
brand_stats["z_value"]  = brand_stats["z_rating"] - brand_stats["z_price"]

# Create price segments
brand_stats['price_segment'] = pd.cut(
    brand_stats['avg_price'],
    bins=[0, 30, 60, 100, 500],
    labels=['Budget\n(<$30)', 'Mid-range\n($30-60)', 'Premium\n($60-100)', 'Luxury\n(>$100)']
)

# ================= Statistical models =================
from scipy import stats
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

# Simple correlation
pearson_r, pearson_p = stats.pearsonr(brand_stats["avg_price"], brand_stats["avg_rating"])

# Weighted regression
x = brand_stats["avg_price"].to_numpy()
y = brand_stats["avg_rating"].to_numpy()
w = brand_stats["reviews_n"].astype(float).clip(lower=1.0).to_numpy()

X = sm.add_constant(x)
wls_model = sm.WLS(y, X, weights=w).fit()

# Get slope and confidence interval
b  = float(wls_model.params[1])
se = float(wls_model.bse[1])
crit = stats.t.ppf(0.975, df=wls_model.df_resid)
slope_w = b
slope_ci = [b - crit*se, b + crit*se]

# Create prediction grid
x_grid = np.linspace(x.min(), x.max(), 200)
Xg = sm.add_constant(x_grid)
yg_wls = wls_model.predict(Xg)

# Bootstrap confidence band
B = 1500
idx = rng.integers(0, len(brand_stats), (B, len(brand_stats)))
Yhat = []
for ii in idx:
    sub = brand_stats.iloc[ii]
    Xb = sm.add_constant(sub["avg_price"])
    yb = sub["avg_rating"]
    wb = sub["reviews_n"].astype(float).clip(lower=1.0)
    try:
        m = sm.WLS(yb, Xb, weights=wb).fit()
        Yhat.append(m.predict(Xg))
    except Exception:
        pass
Yhat = np.vstack(Yhat)
y_lo, y_hi = np.quantile(Yhat, [0.025, 0.975], axis=0)

# Smooth trend line
low = lowess(brand_stats["avg_rating"], brand_stats["avg_price"], frac=0.35, it=1, return_sorted=True)

# Quantile regression lines
qr = sm.QuantReg(brand_stats["avg_rating"], sm.add_constant(brand_stats["avg_price"]))
q_lines = []
for tau, lw, ls, lab in [(0.50, 1.8, ":", "Quantile 0.50"),
                         (0.25, 1.0, ":", "Quantile 0.25"),
                         (0.75, 1.0, ":", "Quantile 0.75")]:
    qfit = qr.fit(q=tau)
    yq = qfit.predict(Xg)
    q_lines.append((yq, lw, ls, lab))

# ================= Create output folder =================
os.makedirs(OUT_DIR, exist_ok=True)

# ================= Figure 1: Main scatter plot =================
print("\nGenerating Figure 1: Main scatter plot...")
fig1, ax = plt.subplots(figsize=(10, 7), facecolor="white")

# Scatter plot with size based on review count
sizes = np.sqrt(brand_stats["reviews_n"]) * 10
ax.scatter(
    brand_stats["avg_price"], brand_stats["avg_rating"],
    s=sizes, c=BARBIE_PINK, alpha=0.75,
    edgecolors=BLACK, linewidth=0.8, zorder=3
)

# Add reference lines at means
mx = brand_stats["avg_price"].mean()
my = brand_stats["avg_rating"].mean()
ax.axvline(mx, color=LIGHT_PINK, linestyle="--", linewidth=1.2, alpha=0.95, zorder=0)
ax.axhline(my, color=LIGHT_PINK, linestyle="--", linewidth=1.2, alpha=0.95, zorder=0)
ax.text(mx*0.98, my*1.01, "Best Value", ha="right", va="bottom", fontsize=10, color=CHARCOAL)

# Add regression lines and confidence band
ax.fill_between(x_grid, y_lo, y_hi, color=LIGHT_PINK, alpha=0.25, linewidth=0, zorder=1, label="WLS 95% band")
ax.plot(x_grid, yg_wls, color=BLACK, linestyle="--", linewidth=2.0, alpha=0.95, label="Weighted regression")
ax.plot(low[:,0], low[:,1], color=ACCENT, linewidth=2.0, alpha=0.95, label="LOWESS")
for yq, lw, ls, lab in q_lines:
    ax.plot(x_grid, yq, color=CHARCOAL, linewidth=lw, linestyle=ls, alpha=0.8, label=lab)

# Label interesting brands
label_candidates = pd.concat([
    brand_stats.nlargest(4, "z_value"),
    brand_stats.nlargest(2, "reviews_n"),
    brand_stats.nsmallest(2, "avg_price"),
    brand_stats.nlargest(2, "avg_price")
]).drop_duplicates("brand")
texts = []
for _, row in label_candidates.iterrows():
    texts.append(ax.text(row["avg_price"], row["avg_rating"], row["brand"], fontsize=9, color=CHARCOAL))
adjust_text = try_import_adjusttext()
if adjust_text and len(texts) > 0:
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.5, alpha=0.6))

# Format plot
ax.set_xlim(0, min(300, brand_stats["avg_price"].max()*1.05))
ax.set_ylim(brand_stats["avg_rating"].min()-0.1, 5.05)
ax.set_xlabel("Average Price (USD)", fontsize=12)
ax.set_ylabel("Average Rating (1-5)", fontsize=12)
ax.set_title("Sephora Product Value Analysis\nPrice vs Rating by Brand", fontsize=16, pad=15, weight='bold')
ax.legend(loc='lower right', frameon=True, framealpha=0.95, fontsize=9)
style_axes(ax)

# Save figure
plt.tight_layout()
fig1_path = os.path.join(OUT_DIR, "figure1_main_scatter.png")
plt.savefig(fig1_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(fig1_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   Saved: {fig1_path}")

# ================= Figure 2: Price segment pie chart =================
print("\nGenerating Figure 2: Price segment pie chart...")
fig2, ax = plt.subplots(figsize=(8, 6), facecolor="white")

segment_counts = brand_stats['price_segment'].value_counts().sort_index()
colors_pie = GRADIENT_COLORS[:len(segment_counts)]

wedges, texts, autotexts = ax.pie(
    segment_counts.values,
    labels=segment_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors_pie,
    wedgeprops={'edgecolor': 'white', 'linewidth': 3},
    textprops={'fontsize': 11, 'color': CHARCOAL, 'weight': 'bold'}
)

# Make percentage labels white
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(11)
    autotext.set_weight('bold')

ax.set_title("Price Segment Distribution\nAcross Beauty Brands", fontsize=16, pad=20, weight='bold')

# Save figure
plt.tight_layout()
fig2_path = os.path.join(OUT_DIR, "figure2_price_segments.png")
plt.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(fig2_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   Saved: {fig2_path}")

# ================= Figure 3: Reviews vs Value scatter =================
print("\nGenerating Figure 3: Reviews vs Value scatter...")
fig3, ax = plt.subplots(figsize=(9, 7), facecolor="white")

# Create pink-purple gradient colormap
custom_cmap = LinearSegmentedColormap.from_list(
    'pink_purple',
    ['#FFE5F0', '#FFB6D9', '#E0218A', '#B8A4C9', '#7A3E9D']
)

scatter = ax.scatter(
    brand_stats['reviews_n'],
    brand_stats['value'],
    s=120,
    c=brand_stats['avg_price'],
    cmap=custom_cmap,
    alpha=0.8,
    edgecolors=BLACK,
    linewidth=1.0,
    vmin=brand_stats['avg_price'].quantile(0.05),
    vmax=brand_stats['avg_price'].quantile(0.95)
)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.046)
cbar.set_label('Average Price ($)', fontsize=11, weight='bold')
cbar.ax.tick_params(labelsize=9)

# Add reference lines at medians
median_reviews = brand_stats['reviews_n'].median()
median_value = brand_stats['value'].median()
ax.axvline(median_reviews, color=LIGHT_PINK, linestyle='--', linewidth=1.5, alpha=0.8, label='Median reviews')
ax.axhline(median_value, color=LIGHT_PINK, linestyle='--', linewidth=1.5, alpha=0.8, label='Median value')

# Format plot
ax.set_xlabel("Number of Reviews (log scale)", fontsize=12, weight='bold')
ax.set_ylabel("Value Score (Rating / Price)", fontsize=12, weight='bold')
ax.set_title("Brand Popularity vs Value\nColor indicates price level", fontsize=16, pad=15, weight='bold')
ax.set_xscale('log')
ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
style_axes(ax)

# Save figure
plt.tight_layout()
fig3_path = os.path.join(OUT_DIR, "figure3_reviews_vs_value.png")
plt.savefig(fig3_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(fig3_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   Saved: {fig3_path}")

# ================= Figure 4: Top value brands lollipop chart =================
print("\nGenerating Figure 4: Top value champions...")
fig4, ax = plt.subplots(figsize=(10, 6), facecolor="white")

top_value = brand_stats.nlargest(10, 'value').sort_values('value')
n_brands = len(top_value)

# Create pink-purple gradient for lollipops
colors_lollipop = [plt.cm.RdPu(0.3 + 0.7 * i / n_brands) for i in range(n_brands)]

# Draw lollipop stems
ax.hlines(
    y=range(n_brands),
    xmin=0,
    xmax=top_value['value'].values,
    color=LIGHT_PINK,
    alpha=0.5,
    linewidth=4
)

# Draw lollipop dots
ax.scatter(
    top_value['value'].values,
    range(n_brands),
    s=250,
    c=colors_lollipop,
    alpha=0.9,
    edgecolors=BLACK,
    linewidth=1.8,
    zorder=3
)

# Add value labels and prices
for i, (val, brand, price) in enumerate(zip(top_value['value'].values,
                                             top_value['brand'].values,
                                             top_value['avg_price'].values)):
    ax.text(val + 0.015, i, f'{val:.3f}',
            va='center', ha='left', fontsize=9, color=CHARCOAL, weight='bold')
    ax.text(-0.015, i, f'${price:.0f}',
            va='center', ha='right', fontsize=8, color=CHARCOAL, alpha=0.7)

# Format plot
ax.set_yticks(range(n_brands))
ax.set_yticklabels(top_value['brand'].values, fontsize=10, weight='bold')
ax.set_xlabel("Value Score (Rating / Price)", fontsize=12, weight='bold')
ax.set_title("Top-10 Value Champions\nBest rating-to-price ratio", fontsize=16, pad=15, weight='bold')
ax.set_xlim(-0.05, top_value['value'].max() * 1.12)
style_axes(ax)

# Save figure
plt.tight_layout()
fig4_path = os.path.join(OUT_DIR, "figure4_top_value.png")
plt.savefig(fig4_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(fig4_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   Saved: {fig4_path}")

# ================= Print summary statistics =================
N = len(brand_stats)
avg_all_price = brand_stats["avg_price"].mean()
avg_all_rating = brand_stats["avg_rating"].mean()
most_reviewed = brand_stats.loc[brand_stats["reviews_n"].idxmax()]
best_value = brand_stats.sort_values("value", ascending=False).iloc[0]

print("\n" + "="*70)
print("FINAL PROJECT - KEY STATISTICS")
print("="*70)
print(f"Total brands analyzed: {N} (with >={MIN_BRAND_REVIEWS} reviews)")
print(f"Overall average price: ${avg_all_price:.2f}")
print(f"Overall average rating: {avg_all_rating:.2f}/5.0")
print(f"Pearson correlation (price vs rating): r={pearson_r:.3f}, p={pearson_p:.3g}")
print(f"WLS regression slope: {slope_w:.4f} [95% CI: {slope_ci[0]:.4f}, {slope_ci[1]:.4f}]")
print(f"\nBest value brand: {best_value['brand']}")
print(f"   Value score: {best_value['value']:.4f}")
print(f"   Avg rating: {best_value['avg_rating']:.2f}")
print(f"   Avg price: ${best_value['avg_price']:.2f}")
print(f"\nMost reviewed brand: {most_reviewed['brand']}")
print(f"   Total reviews: {int(most_reviewed['reviews_n']):,}")
print(f"\nAll figures saved to: {OUT_DIR}")
print("   figure1_main_scatter.png (+ PDF)")
print("   figure2_price_segments.png (+ PDF)")
print("   figure3_reviews_vs_value.png (+ PDF)")
print("   figure4_top_value.png (+ PDF)")
print("="*70)
