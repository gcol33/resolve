# RESOLVE Roadmap

**R**elational **E**ncoding via **S**tructured **O**bservation **L**earning with **V**ector **E**mbeddings

## v1.0 (Current)

- Generalized PlotEncoder (hash/embed/onehot/numeric/raw)
- R formula interface with `data` + `obs` pattern
- Separate `top_k` and `bottom_k` selection
- Categorical validation (error if bare variable is categorical)

## v1.1 (Planned)

### Multiple observation tables

Support multiple many-to-one relationships in a single model:

```r
resolve(
  ltv ~ age + hash(product_id, from = purchases) + embed(category, from = browsing),
  data = customers,
  obs = list(purchases = purchase_df, browsing = browse_df),
  by = "customer_id"
)
```

Use cases:
- Customers with purchases AND browsing history
- Patients with diagnoses AND prescriptions AND procedures
- Documents with words AND citations AND authors
- Plots with species AND environmental measurements over time

Implementation notes:
- `from = <table_name>` parameter in `hash()`/`embed()`/`onehot()`
- `obs` accepts named list of data frames
- Each table can have different join key (extend `by` to named vector)
