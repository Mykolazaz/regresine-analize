# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.formula.api import quantreg, ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
import patsy


# %%
data = pd.read_csv("insurance.csv")


# %%
data["charges"] = data["charges"] / 1000


# %%
plt.figure()
plt.hist(data["charges"], bins=30, color="mediumseagreen", alpha=0.8)
plt.xlabel("Gydymo kaina (tūkst. JAV dolerių)", fontsize=14)
plt.ylabel("Dažnis", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()


# %%
q1 = data["charges"].quantile(0.25)
q3 = data["charges"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
charges_outliers = data[(data["charges"] < lower) | (data["charges"] > upper)]
len(charges_outliers)


# %%
plt.figure()
plt.boxplot(data["age"], vert=True, patch_artist=True,
            boxprops=dict(facecolor="slateblue", alpha=0.8), medianprops=dict(color="black"))
plt.xlabel("Amžius", fontsize=14)
plt.xticks([])
plt.tight_layout()
plt.show()


# %%
q1 = data["age"].quantile(0.25)
q3 = data["age"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
age_outliers = data[(data["age"] < lower) | (data["age"] > upper)]
len(age_outliers)


# %%
plt.figure()
plt.scatter(data["age"], data["charges"], color="blue", alpha=0.6, s=20)
plt.xlabel("Amžius", fontsize=14)
plt.ylabel("Gydymo kaina (tūkst. JAV dolerių)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("kaina_amzius_scatter.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
data["sex"].value_counts()


# %%
data["sex"].value_counts(normalize=True)


# %%
sex_counts = data["sex"].value_counts().reindex(["female", "male"])

plt.figure()
plt.bar(["Moteris", "Vyras"], sex_counts.values, color=["pink", "lightblue"])
plt.xlabel("Lytis")
plt.ylabel("Dažnis")
plt.yticks(np.arange(0, 1001, 50))
plt.tight_layout()
plt.show()


# %%
female_charges = data.loc[data["sex"] == "female", "charges"]
male_charges = data.loc[data["sex"] == "male", "charges"]

plt.figure()
parts = plt.violinplot([female_charges, male_charges], positions=[1, 2],showextrema=False)
for body, color in zip(parts["bodies"], ["salmon", "steelblue"]):
    body.set_facecolor(color)
    body.set_alpha(0.6)

bp = plt.boxplot([female_charges, male_charges], positions=[1, 2], widths=0.15, patch_artist=True,
            medianprops=dict(color="black"))

for patch, color in zip(bp['boxes'], ["salmon", "steelblue"]):
    patch.set_facecolor(color)

plt.xticks([1, 2], ["Moteris", "Vyras"], fontsize=14)
plt.ylabel("Gydymo kaina (tūkst. JAV dolerių)", fontsize=14)
plt.yticks(np.arange(0, 81, 5), fontsize=14)
plt.tight_layout()
plt.show()


# %%
plt.figure()
plt.hist(data["bmi"], bins=30, color="steelblue", alpha=0.85)
plt.xlabel("KMI", fontsize=14)
plt.ylabel("Dažnis", fontsize=14)
plt.yticks(np.arange(0, 121, 10), fontsize=12)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.show()


# %%
plt.figure()
plt.boxplot(data["bmi"], vert=True, patch_artist=True,
            boxprops=dict(facecolor="steelblue", alpha=0.8), 
            medianprops=dict(color="black"))
plt.xlabel("KMI", fontsize=14)
plt.xticks([])
plt.tight_layout()
plt.show()


# %%
q1 = data["bmi"].quantile(0.25)
q3 = data["bmi"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
bmi_outliers = data[(data["bmi"] < lower) | (data["bmi"] > upper)]
len(bmi_outliers)


# %%
plt.figure()
plt.scatter(data["bmi"], data["charges"], color="steelblue", alpha=0.8, s=20)
plt.xlabel("KMI", fontsize=14)
plt.ylabel("Gydymo kaina (tūkst. JAV dolerių)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("bmi_charges_scatter.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
data["children"].astype("category").value_counts().sort_index()


# %%
data["children"].astype("category").value_counts(normalize=True).sort_index()


# %%
children_levels = sorted(data["children"].unique())
children_data = [data.loc[data["children"] == c, "charges"] for c in children_levels]

plt.figure()
parts = plt.violinplot(children_data, positions=range(1, len(children_levels) + 1),
                       showmeans=False, showmedians=False, showextrema=False)
for body in parts["bodies"]:
    body.set_facecolor("mediumpurple")
    body.set_alpha(0.6)

plt.boxplot(children_data, positions=range(1, len(children_levels) + 1), widths=0.15,
            patch_artist=True, boxprops=dict(facecolor="mediumpurple"),
            medianprops=dict(color="black"))

plt.xticks(range(1, len(children_levels) + 1), children_levels, fontsize=12)
plt.xlabel("Vaikų skaičius", fontsize=14)
plt.ylabel("Gydymo kaina (tūkst. JAV dolerių)", fontsize=14)
plt.yticks(np.arange(0, 81, 10), fontsize=12)
plt.tight_layout()
plt.savefig("vaikai_charges_violin.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
data["smoker"].value_counts()


# %%
data["smoker"].value_counts(normalize=True)


# %%
smoker_yes = data.loc[data["smoker"] == "yes", "charges"]
smoker_no = data.loc[data["smoker"] == "no", "charges"]

plt.figure()
parts = plt.violinplot([smoker_yes, smoker_no], positions=[1, 2], showmeans=False, showmedians=False, showextrema=False)
for body, color in zip(parts["bodies"], ["mediumseagreen", "indianred"]):
    body.set_facecolor(color)
    body.set_alpha(0.6)

bp=plt.boxplot([smoker_yes, smoker_no], positions=[1, 2], widths=0.15, patch_artist=True,
               medianprops=dict(color="black"))
for patch, color in zip(bp['boxes'], ["mediumseagreen", "indianred"]):
	patch.set_facecolor(color)
plt.xticks([1, 2], ["Rūko", "Nerūko"], fontsize=12)
plt.ylabel("Gydymo kaina (tūkst. JAV dolerių)", fontsize=14)
plt.yticks(np.arange(0, 81, 10), fontsize=12)
plt.tight_layout()
plt.show()


# %%
data["region"].value_counts()


# %%
data["region"].value_counts(normalize=True)


# %%
region_order = ["northeast", "northwest", "southeast", "southwest"]
region_labels = ["Šiaurės rytai", "Šiaurės vakarai", "Pietų rytai", "Pietų vakarai"]
region_counts = data["region"].value_counts().reindex(region_order)

plt.figure()
plt.bar(region_labels, region_counts.values, color="darkorange", alpha=0.8)
plt.ylabel("Dažnis", fontsize=14)
plt.yticks(np.arange(0, 401, 50), fontsize=12)
plt.xticks(fontsize=12, rotation=0)
plt.tight_layout()
plt.show()


# %%
region_data = [data.loc[data["region"] == r, "charges"] for r in region_order]

plt.figure()
parts = plt.violinplot(region_data, positions=range(1, len(region_order) + 1),
                       showmeans=False, showmedians=False, showextrema=False)
for body in parts["bodies"]:
    body.set_facecolor("darkorange")
    body.set_alpha(0.6)

plt.boxplot(region_data, positions=range(1, len(region_order) + 1), widths=0.15,
            patch_artist=True, boxprops=dict(facecolor="darkorange", alpha=0.8),
            medianprops=dict(color="black"))

plt.xticks(range(1, len(region_order) + 1), region_labels, fontsize=12)
plt.ylabel("Gydymo kaina (tūkst. JAV dolerių)", fontsize=14)
plt.yticks(np.arange(0, 81, 10), fontsize=12)
plt.tight_layout()
plt.savefig("regionai_charges_violin.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
q75 = data["charges"].quantile(0.75)

mod_q75_bmi = quantreg("charges ~ bmi", data).fit(q=0.75)

x_grid = np.linspace(data["bmi"].min(), data["bmi"].max(), 200)
pred_df = pd.DataFrame({"bmi": x_grid})
y_pred = mod_q75_bmi.predict(pred_df)

plt.figure()
plt.scatter(data["bmi"], data["charges"], alpha=0.3)
plt.plot(x_grid, y_pred, color="red", linewidth=1.2)
plt.ylabel("Charges")
plt.title("Predicted 75th percentile of charges by BMI")
plt.tight_layout()
plt.show()


# %%
q75_region = (
    data.groupby("region", as_index=False)["charges"]
    .quantile(0.75)
    .rename(columns={"charges": "q75"})
)
q75_region


# %%
corr_matrix = data[["age", "bmi", "children", "charges"]].corr(method="spearman")
corr_matrix.index = ["Amžius", "KMI", "Vaikai", "Gydymo kaina"]
corr_matrix.columns = ["Amžius", "KMI", "Vaikai", "Gydymo kaina"]

plt.figure(figsize=(6, 6))
plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha="right")
plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)

for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black")

plt.tight_layout()
plt.show()


# %%
X_num = data[["age", "bmi", "children"]].copy()
X_num["intercept"] = 1

vif_data = pd.DataFrame({
    "variable": X_num.columns,
    "VIF": [variance_inflation_factor(X_num.values, i) for i in range(X_num.shape[1])]
})

print(vif_data)


# %%
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)


# %%
model1 = quantreg("charges ~ age + bmi + children + C(sex) + C(smoker) + C(region)", data).fit(q=0.75)


# %%
print(model1.summary())


# %%
model2 = quantreg("charges ~ age + bmi + children + C(sex) + C(smoker) + C(region) + age:bmi", data).fit(q=0.75)


# %%
print(model2.summary())


# %%
model3 = quantreg("charges ~ age + bmi + children + C(sex) + C(smoker) + C(region) + age:bmi + C(smoker):bmi", data).fit(q=0.75)


# %%
print(model3.summary())


# %%
taus = np.arange(0.1, 0.91, 0.05)

coefs_age = []
coefs_bmi = []
coefs_children = []

for tau in taus:
    m = quantreg("charges ~ age + bmi + children + C(sex) + C(smoker) + C(region)", data).fit(q=tau)
    coefs_age.append(m.params.get("age", np.nan))
    coefs_bmi.append(m.params.get("bmi", np.nan))
    coefs_children.append(m.params.get("children", np.nan))


# %%
plt.figure()
plt.plot(taus, coefs_age)
plt.xlabel("Tau")
plt.ylabel("age koeficientas")
plt.yticks(np.arange(0.2, 0.32, 0.02))
plt.title("age")
plt.show()


# %%
plt.figure()
plt.plot(taus, coefs_bmi)
plt.xlabel("Tau")
plt.ylabel("bmi koeficientas")
plt.yticks(np.arange(0, 0.61, 0.2))
plt.title("bmi")
plt.show()


# %%
plt.figure()
plt.plot(taus, coefs_children)
plt.xlabel("Tau")
plt.ylabel("children koeficientas")
plt.yticks(np.arange(0, 1.1, 0.5))
plt.title("children")
plt.show()


# %%
fit0 = quantreg("charges ~ 1", train_data).fit(q=0.75)


def rho(u, tau=0.75):
    return u * (tau - (u < 0).astype(int))

model1_train = quantreg("charges ~ age + bmi + children + C(sex) + C(smoker) + C(region)", train_data).fit(q=0.75)
model3_train = quantreg("charges ~ age + bmi + children + C(sex) + C(smoker) + C(region) + age:bmi + C(smoker):bmi", train_data).fit(q=0.75)

rho_fit0 = rho(train_data["charges"] - fit0.predict(train_data), tau=0.75).sum()
rho_model1 = rho(train_data["charges"] - model1_train.predict(train_data), tau=0.75).sum()
rho_model3 = rho(train_data["charges"] - model3_train.predict(train_data), tau=0.75).sum()

R1 = 1 - rho_model1 / rho_fit0
R2 = 1 - rho_model3 / rho_fit0

R1, R2



