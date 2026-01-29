# Round 01 Plot Guide (Internal)

This is an internal, novice-friendly guide to explain what the PCA and scatter plots show for each function.
It stays descriptive: ranges, variance structure, and which dimensions are most associated with y in the current data.

## How to read the plots (quick primer)

- Scatter x1 vs x2 (colored by y): shows coverage in 2D; color gradients indicate where y is higher or lower.
- Dim vs y scatter: one plot per dimension; a visible slope suggests that dimension may influence y.
- PCA (2D/3D): compresses inputs into principal components; the % variance tells you how much of the input spread is captured.
- Correlation heatmap: linear association between each x and y (positive/negative).
- Parallel coordinates: all dimensions on one chart; color highlights whether higher y aligns with certain ranges.

## Function 1

- Samples: 10, Dimensions: 2
- y range: -0.0036 to 0.0000 (mean -0.0004, std 0.0011)
- x ranges: x1 0.0825 to 0.8839, x2 0.0787 to 0.8799
- PCA variance (PC1/PC2): 66.5%, 33.5%
- Strongest linear association with y (by |corr|): x2 (corr -0.169)
- Plots to review: plots/function_1/scatter_x1_x2.png, plots/function_1/dim_vs_y.png, plots/function_1/pca_2d.png

Suggested talking points:
- Use the x-range line to show the spread of sampled inputs; a narrow range implies limited coverage in that dimension.
- Use PCA variance to explain whether the inputs lie mostly along a few directions (high PC1/PC2 %) or are more evenly spread.
- Use the correlation note as a cautious, linear-only hint; it is not proof of causality or the global shape.

## Function 2

- Samples: 10, Dimensions: 2
- y range: -0.0656 to 0.6112 (mean 0.2307, std 0.2254)
- x ranges: x1 0.1427 to 0.8778, x2 0.0287 to 0.9266
- PCA variance (PC1/PC2): 81.6%, 18.4%
- Strongest linear association with y (by |corr|): x1 (corr 0.752)
- Plots to review: plots/function_2/scatter_x1_x2.png, plots/function_2/dim_vs_y.png, plots/function_2/pca_2d.png

Suggested talking points:
- Use the x-range line to show the spread of sampled inputs; a narrow range implies limited coverage in that dimension.
- Use PCA variance to explain whether the inputs lie mostly along a few directions (high PC1/PC2 %) or are more evenly spread.
- Use the correlation note as a cautious, linear-only hint; it is not proof of causality or the global shape.

## Function 3

- Samples: 15, Dimensions: 3
- y range: -0.3989 to -0.0348 (mean -0.1072, std 0.0842)
- x ranges: x1 0.0468 to 0.9660, x2 0.2199 to 0.9414, x3 0.0661 to 0.9909
- PCA variance (PC1/PC2/PC3): 51.2%, 32.4%, 16.3%
- Strongest linear association with y (by |corr|): x3 (corr -0.574)
- Plots to review: plots/function_3/scatter_x1_x2.png, plots/function_3/dim_vs_y.png, plots/function_3/pca_2d.png

Suggested talking points:
- Use the x-range line to show the spread of sampled inputs; a narrow range implies limited coverage in that dimension.
- Use PCA variance to explain whether the inputs lie mostly along a few directions (high PC1/PC2 %) or are more evenly spread.
- Use the correlation note as a cautious, linear-only hint; it is not proof of causality or the global shape.

## Function 4

- Samples: 30, Dimensions: 4
- y range: -32.6257 to -4.0255 (mean -17.2386, std 7.0180)
- x ranges: x1 0.0378 to 0.9856, x2 0.0063 to 0.9196, x3 0.0422 to 0.9392, x4 0.0815 to 0.9995
- PCA variance (PC1/PC2/PC3): 37.9%, 27.3%, 21.5%
- Strongest linear association with y (by |corr|): x1 (corr -0.540)
- Plots to review: plots/function_4/scatter_x1_x2.png, plots/function_4/dim_vs_y.png, plots/function_4/pca_2d.png

Suggested talking points:
- Use the x-range line to show the spread of sampled inputs; a narrow range implies limited coverage in that dimension.
- Use PCA variance to explain whether the inputs lie mostly along a few directions (high PC1/PC2 %) or are more evenly spread.
- Use the correlation note as a cautious, linear-only hint; it is not proof of causality or the global shape.

## Function 5

- Samples: 20, Dimensions: 4
- y range: 0.1129 to 1088.8596 (mean 151.2719, std 245.5760)
- x ranges: x1 0.1199 to 0.8365, x2 0.0382 to 0.8625, x3 0.0889 to 0.8795, x4 0.0729 to 0.9576
- PCA variance (PC1/PC2/PC3): 39.6%, 25.9%, 21.6%
- Strongest linear association with y (by |corr|): x4 (corr 0.570)
- Plots to review: plots/function_5/scatter_x1_x2.png, plots/function_5/dim_vs_y.png, plots/function_5/pca_2d.png

Suggested talking points:
- Use the x-range line to show the spread of sampled inputs; a narrow range implies limited coverage in that dimension.
- Use PCA variance to explain whether the inputs lie mostly along a few directions (high PC1/PC2 %) or are more evenly spread.
- Use the correlation note as a cautious, linear-only hint; it is not proof of causality or the global shape.

## Function 6

- Samples: 20, Dimensions: 5
- y range: -2.5712 to -0.7143 (mean -1.4954, std 0.4490)
- x ranges: x1 0.0217 to 0.9577, x2 0.1144 to 0.9319, x3 0.0165 to 0.9788, x4 0.0456 to 0.9617, x5 0.0049 to 0.8928
- PCA variance (PC1/PC2/PC3): 34.9%, 27.7%, 15.8%
- Strongest linear association with y (by |corr|): x5 (corr -0.584)
- Plots to review: plots/function_6/scatter_x1_x2.png, plots/function_6/dim_vs_y.png, plots/function_6/pca_2d.png

Suggested talking points:
- Use the x-range line to show the spread of sampled inputs; a narrow range implies limited coverage in that dimension.
- Use PCA variance to explain whether the inputs lie mostly along a few directions (high PC1/PC2 %) or are more evenly spread.
- Use the correlation note as a cautious, linear-only hint; it is not proof of causality or the global shape.

## Function 7

- Samples: 30, Dimensions: 6
- y range: 0.0027 to 1.3650 (mean 0.2196, std 0.3021)
- x ranges: x1 0.0579 to 0.9425, x2 0.0118 to 0.9247, x3 0.0036 to 0.9246, x4 0.0737 to 0.9610, x5 0.0149 to 0.9987, x6 0.0511 to 0.9510
- PCA variance (PC1/PC2/PC3): 28.3%, 25.7%, 17.5%
- Strongest linear association with y (by |corr|): x5 (corr -0.378)
- Plots to review: plots/function_7/scatter_x1_x2.png, plots/function_7/dim_vs_y.png, plots/function_7/pca_2d.png

Suggested talking points:
- Use the x-range line to show the spread of sampled inputs; a narrow range implies limited coverage in that dimension.
- Use PCA variance to explain whether the inputs lie mostly along a few directions (high PC1/PC2 %) or are more evenly spread.
- Use the correlation note as a cautious, linear-only hint; it is not proof of causality or the global shape.

## Function 8

- Samples: 40, Dimensions: 8
- y range: 5.5922 to 9.5985 (mean 7.8153, std 0.9469)
- x ranges: x1 0.0091 to 0.9859, x2 0.0034 to 0.9740, x3 0.0229 to 0.9989, x4 0.0090 to 0.9030, x5 0.0096 to 0.9869, x6 0.0221 to 0.9902, x7 0.0359 to 0.9929, x8 0.0420 to 0.9888
- PCA variance (PC1/PC2/PC3): 20.3%, 20.1%, 16.2%
- Strongest linear association with y (by |corr|): x1 (corr -0.626)
- Plots to review: plots/function_8/scatter_x1_x2.png, plots/function_8/dim_vs_y.png, plots/function_8/pca_2d.png

Suggested talking points:
- Use the x-range line to show the spread of sampled inputs; a narrow range implies limited coverage in that dimension.
- Use PCA variance to explain whether the inputs lie mostly along a few directions (high PC1/PC2 %) or are more evenly spread.
- Use the correlation note as a cautious, linear-only hint; it is not proof of causality or the global shape.
