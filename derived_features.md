# Derived features

For each forecast-origin scenario k = 1 to 5, the workflow derives:

- observed prefix milk-yield records;
- observed prefix DIM records;
- parity-by-record prior for each record;
- observed prefix residuals = observed milk yield minus fold-specific prior;
- mean prefix residual;
- last prefix residual;
- residual slope;
- prefix mean;
- prefix maximum;
- last observed prefix milk yield;
- early rise;
- raw milk-yield slope;
- prior prefix mean;
- prior prefix last value;
- target record number;
- target DIM or record-mean target-DIM approximation.

Future records, post-origin covariates, and validation-fold-derived preprocessing values must not enter model fitting.
